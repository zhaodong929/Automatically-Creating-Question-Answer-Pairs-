import scrapy
from usda.items import UsdaSpiderItem
from urllib.parse import urlparse
from w3lib.html import remove_tags
import hashlib
import re

class UsdacrawlSpider(scrapy.Spider):
    name = "usdacrawl"
    allowed_domains = ["usda.gov"]
    start_urls = ["https://www.usda.gov/topics"]

    visited_urls = set()
    theme_links = []  # 用于存储所有主题的链接
    current_theme_index = 0  # 当前处理的主题索引

    def parse(self, response):
        # 提取所有主题链接并存储到 theme_links 列表中
        themes = response.css('div.usa-list-items-wrapper > div.usa-width-one-third')
        self.log(f"Found {len(themes)} themes.")
        
        for theme in themes:
            theme_name = theme.css('h3 a::text').get().strip()
            theme_link = theme.css('h3 a::attr(href)').get()
            self.theme_links.append((theme_name, theme_link))
        
        # 开始爬取第一个主题
        if self.theme_links:
            self.log(f"Starting to scrape theme: {self.theme_links[self.current_theme_index][0]}")
            yield response.follow(self.theme_links[self.current_theme_index][1], self.parse_page, meta={'theme': self.theme_links[self.current_theme_index][0], 'sub_theme': 'Main Page', 'level': 1, 'parent_link': response.url})

    def parse_page(self, response):
        theme_name = response.meta['theme']
        sub_theme = response.meta.get('sub_theme', 'Sub Page')
        level = response.meta['level']
        parent_link = response.meta['parent_link']

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

        clean_content = remove_tags(body_content).strip()

        # 去掉特定的无关内容，例如 "Skip to main content" 等
        unwanted_phrases = ["Skip to main content", "Pagination", "Last page", "Next page", "Return to top", "An official website of the United States government"]
        for phrase in unwanted_phrases:
            clean_content = clean_content.replace(phrase, '')

        # 使用正则表达式去除多余的换行符
        clean_content = re.sub(r'\n+', '\n', clean_content).strip()

        # 过滤掉空白或无效内容
        if len(clean_content) < 100:
            self.log(f"Content too short, skipping page: {response.url}")
            return

        self.log(f"Scraping page: {response.url} for theme: {theme_name}")

        yield UsdaSpiderItem(
            theme=theme_name,
            sub_theme=sub_theme,
            content=clean_content,
            link=response.url,
            level=level,
            parent_link=parent_link
        )

        # 继续获取当前页面所有链接并递归爬取
        links = response.css('a::attr(href)').getall()
        self.log(f"Found {len(links)} links on page {response.url}")
        for link in links:
            parsed_url = urlparse(link)
            if parsed_url.netloc in ['', 'www.usda.gov'] and not self.is_irrelevant_link(link):
                yield response.follow(link, self.parse_page, meta={'theme': theme_name, 'sub_theme': self.extract_sub_theme(link), 'level': level + 1, 'parent_link': response.url})

        # 如果该主题爬取完成，检查是否有下一个主题
        if level == 1:  # 确保只有在顶层页面返回时才进行下一个主题的爬取
            self.current_theme_index += 1
            if self.current_theme_index < len(self.theme_links):
                next_theme_name, next_theme_link = self.theme_links[self.current_theme_index]
                self.log(f"Moving to next theme: {next_theme_name}")
                yield response.follow(next_theme_link, self.parse_page, meta={'theme': next_theme_name, 'sub_theme': 'Main Page', 'level': 1, 'parent_link': response.url})

    def extract_sub_theme(self, link):
        parts = urlparse(link).path.strip('/').split('/')
        if parts:
            return parts[-1].replace('-', ' ').capitalize()
        return 'Sub Page'

    def is_irrelevant_link(self, link):
        irrelevant_keywords = ['facebook', 'twitter', 'linkedin', 'contact', 'privacy', 'sitemap', 'login', 'register', 'about', 'help']
        return any(keyword in link for keyword in irrelevant_keywords)
