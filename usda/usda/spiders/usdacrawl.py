import scrapy
from usda.items import UsdaSpiderItem
from w3lib.html import remove_tags
import hashlib
import re
from urllib.parse import urlparse

class UsdacrawlSpider(scrapy.Spider):
    name = "usdacrawl"
    allowed_domains = ["usda.gov"]
    start_urls = ["https://www.usda.gov/topics"]

    visited_urls = set()
    theme_links = []  # 用于存储所有主题的链接
    current_theme_index = 16  # 从第三个主题（索引2）开始处理
    max_items = 5000  # 设置从第三个板块开始的数据条数限制
    items_count = 0  # 计数器，记录当前爬取的数据条数

    def parse(self, response):
        # 提取所有主题链接并存储到 theme_links 列表中
        themes = response.css('div.usa-list-items-wrapper > div.usa-width-one-third')
        self.log(f"Found {len(themes)} themes.")

        for theme in themes:
            theme_name = theme.css('h3 a::text').get().strip()
            theme_link = theme.css('h3 a::attr(href)').get()
            self.theme_links.append((theme_name, theme_link))

        # 从第三个主题开始爬取
        if len(self.theme_links) > 16:  # 确保有多个板块可以跳过前两个
            self.log(f"Starting to scrape theme: {self.theme_links[self.current_theme_index][0]}")
            yield response.follow(self.theme_links[self.current_theme_index][1], self.parse_page, meta={'theme': self.theme_links[self.current_theme_index][0]})

    def parse_page(self, response):
        # 确保从 meta 中正确获取主题名称
        theme_name = response.meta.get('theme', 'Unknown')

        # 使用哈希避免重复访问相同的URL
        url_hash = hashlib.md5(response.url.encode('utf-8')).hexdigest()
        if url_hash in self.visited_urls:
            self.log(f"Skipping already visited URL: {response.url}")
            return
        self.visited_urls.add(url_hash)

        # 选择主要内容区域，过滤掉常见的无关部分
        body_content = response.css('article, .content, .main-content, .entry-content').get()
        if not body_content:
            self.log(f"No meaningful content found, skipping page: {response.url}")
            return

        # 清理并过滤内容
        clean_content, author = self.clean_text(body_content)

        # 过滤掉空白或无效内容
        if len(clean_content) < 50:  # 调整过滤长度
            self.log(f"Content too short, skipping page: {response.url}")
            return

        self.log(f"Scraping page: {response.url} for theme: {theme_name}")

        # 提取到的数据
        yield UsdaSpiderItem(
            theme=theme_name,
            content=clean_content,
            author=author
        )

        # 更新计数器
        self.items_count += 1

        # 检查是否达到限制
        if self.items_count >= self.max_items:
            self.log("Reached the maximum number of items. Stopping further scraping.")
            # 达到限制后立即跳转到下一个主题
            yield from self.move_to_next_theme(response)
            return

        # 继续获取当前页面所有链接并递归爬取
        links = response.css('a::attr(href)').getall()
        self.log(f"Found {len(links)} links on page {response.url}")
        for link in links:
            parsed_url = urlparse(link)
            if parsed_url.netloc in ['', 'www.usda.gov'] and not self.is_irrelevant_link(link):
                yield response.follow(link, self.parse_page, meta={'theme': theme_name})

    def move_to_next_theme(self, response):
        """
        跳转到下一个主题（板块）。
        """
        # 确保递归到下一个主题
        if self.current_theme_index < len(self.theme_links) - 1:
            self.current_theme_index += 1
            next_theme_name, next_theme_link = self.theme_links[self.current_theme_index]
            self.log(f"Moving to next theme: {next_theme_name}")
            self.items_count = 0  # 重置计数器
            yield response.follow(next_theme_link, self.parse_page, meta={'theme': next_theme_name})
        else:
            self.log("All themes have been scraped.")

    def clean_text(self, text):
        """
        清理文本中的多余换行符、空行、空格，提取作者信息，并去除无关信息如电话号码、地址、文件下载信息、报告数据和JavaScript代码
        """
        # 改进的正则表达式，用于提取符合姓名格式的作者信息，包括中间名缩写
        author_patterns = [
            r'Posted by\s+(Dr\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-zA-Z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-zA-Z]+)+)(,.*?)?\b',  # 匹配带职称和中间名缩写的名字
            r'By\s+(Dr\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-zA-Z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-zA-Z]+)+)(,.*?)?\b',         # 处理 "By" 开头的格式
        ]

        author = "Unknown"
        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                # 提取全称，包含职称、名字和中间名缩写
                author = match.group(2).strip() if match.group(2) else "Unknown"
                # 移除匹配到的完整内容，包括尾随的无关描述
                text = re.sub(re.escape(match.group(0)), '', text)
                break

        # 去除 HTML 标签
        text = remove_tags(text).strip()

        # 替换特殊的 HTML 实体，如 &nbsp;
        html_entities = {
            '&nbsp;': ' ',    # 不间断空格替换为空格
            '&amp;': '&',     # 替换 & 符号
            '&quot;': '"',    # 替换双引号
            '&lt;': '<',      # 替换小于号
            '&gt;': '>',      # 替换大于号
            # 可以根据需要扩展其他 HTML 实体
        }

        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)

        # 去除 CSS 样式和脚本内容
        text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)  # 去除 CSS
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)  # 去除 JavaScript

        # 去除特定无关的 JavaScript 函数调用，如 adobeDCView
        text = re.sub(r'adobeDCView\.previewFile\([^)]*\);?', '', text, flags=re.DOTALL | re.IGNORECASE)

        # 去除 JavaScript 事件监听和函数调用等内容
        text = re.sub(r'\b(document|window)\.addEventListener\([^\)]+\);?', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\b(function\s*\w*\([^)]*\)\s*{[^}]*})', '', text, flags=re.DOTALL | re.IGNORECASE)  # 去除函数定义

        # 去除 CSS 内联样式或其他代码片段
        text = re.sub(r'\.\w+[\w\s-]*{[^}]*}', '', text, flags=re.DOTALL)  # 去除 CSS 类选择器样式
        text = re.sub(r'[{].*?[}]', '', text, flags=re.DOTALL)  # 处理所有花括号包围的内容

        # 去除无效的文件下载信息、报告名称、年份等无关内容
        unwanted_text_patterns = [
            r'\b(Secondary Navigation|DIGITAL STRATEGY|Design and Brand|Web Design System)\b',  # 去除设计类无关标题
            r'\b(NBSP)+\b',  # 去除重复的 NBSP 无用内容
            r'\b(fonts|colors|icons|guidelines|required|reports|data|FY|quarterly)\b',  # 去除常见的设计术语和报告名称
            r'\.\w+[-\w]*:hover\b',  # 去除 hover 类选择器
            r'\#\w+[-\w]*\b',  # 去除 ID 选择器
            r'\(\s*,\s*\d+(\.\d+)?\s*\)',  # 精确删除括号内含逗号和数字的无效内容
            r'[(]\s*\d+[\w\s.,-]*\)',  # 删除括号内包含逗号和数字的无效内容
            r'\d{4}',  # 去除年份（例如2023、2022等）
            r'\b(MB|KB|GB)\b',  # 去除文件大小
            r'\bImplementation Updates\s*[\d\.-]+',  # 去除Implementation Updates及其后续版本号
            r'https?://\S+',  # 去除网址链接
        ]
        for pattern in unwanted_text_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 去除包含文件名称和报告相关的内容
        text = re.sub(r'(XML|XLSX|CSV|PDF|DOCX|TXT|PPTX|PPT)\b', '', text, flags=re.IGNORECASE)

        # 去除电话号码
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)

        # 去除地址（示例：使用常见地址格式）
        text = re.sub(r'\d{1,5} [A-Za-z0-9 ]+ (Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr)', '', text)

        # 去除特定无关短语
        unwanted_phrases = [
            "Return to top", "Last page", "Next page", "Skip to main content",
            "An official website of the United States government", "PDF", "KB"
        ]
        for phrase in unwanted_phrases:
            text = text.replace(phrase, '')

        # 去除多余的换行符和空行
        text = re.sub(r'\n\s*\n+', '\n', text)

        # 替换多余的空格
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip(), author

    def is_irrelevant_link(self, link):
        irrelevant_keywords = ['facebook', 'twitter', 'linkedin', 'contact', 'privacy', 'sitemap', 'login', 'register', 'about', 'help']
        return any(keyword in link for keyword in irrelevant_keywords)
