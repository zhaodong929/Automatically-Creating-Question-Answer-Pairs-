import json
import os
from itemadapter import ItemAdapter

class UsdaPipeline:
    def open_spider(self, spider):
        # 确保保存文件的目录存在
        self.save_dir = 'output'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 打开文件流字典
        self.files = {}

    def close_spider(self, spider):
        # 关闭所有打开的文件
        for file in self.files.values():
            self.close_file(file)

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        theme = adapter.get('theme')
        
        if theme not in self.files:
            # 为每个主题创建一个新的JSON文件
            file_path = os.path.join(self.save_dir, f"{theme}.json")
            self.files[theme] = open(file_path, 'w', encoding='utf-8')
            self.files[theme].write('[')  # 开始写入JSON数组
            spider.log(f"Creating new file for theme: {theme}")

        # 将item转换为JSON格式，并写入对应的文件中
        line = json.dumps(adapter.asdict(), ensure_ascii=False)
        self.files[theme].write(line + ',\n')
        spider.log(f"Writing item to file for theme: {theme}")

        return item

    def close_file(self, file):
        # 移除最后一个逗号，并关闭JSON数组和文件
        file.seek(file.tell() - 2, os.SEEK_SET)
        file.write('\n]')
        file.close()
