import json
import os
import mysql.connector
from itemadapter import ItemAdapter

class UsdaPipeline:
    def open_spider(self, spider):
        # 确保保存文件的目录存在
        self.save_dir = 'output'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 打开文件流字典
        self.files = {}
        self.file_paths = {}

        # 创建 MySQL 数据库连接
        try:
            self.conn = mysql.connector.connect(
                host='localhost',           # MySQL服务器地址
                user='root',       # MySQL用户名
                password='123456',   # MySQL密码
                database='usda_data'        # 数据库名称
            )
            self.cursor = self.conn.cursor()

            # 创建表格，如果不存在则创建
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS themes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    theme VARCHAR(255),
                    content TEXT,
                    author VARCHAR(255)
                )
            ''')
            self.conn.commit()
            spider.log("Connected to MySQL database.")
        except mysql.connector.Error as err:
            spider.log(f"Error connecting to MySQL: {err}")

    def close_spider(self, spider):
        # 关闭所有打开的文件
        for theme, file in self.files.items():
            try:
                self.close_file(file, self.file_paths[theme])
            except Exception as e:
                spider.log(f"Error closing file for theme {theme}: {e}")

        # 关闭 MySQL 连接
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            spider.log("MySQL connection closed.")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        theme = adapter.get('theme', 'Unknown').replace('/', '_')  # 确保主题名称安全用于文件名

        # JSON 文件写入部分
        if theme not in self.files:
            try:
                # 为每个主题创建一个新的 JSON 文件
                file_path = os.path.join(self.save_dir, f"{theme}.json")
                self.file_paths[theme] = file_path
                self.files[theme] = open(file_path, 'w', encoding='utf-8')
                self.files[theme].write('[')  # 开始写入 JSON 数组
                spider.log(f"Creating new file for theme: {theme}")
            except Exception as e:
                spider.log(f"Error creating file for theme {theme}: {e}")
                return item

        try:
            # 将 item 转换为 JSON 格式，并写入对应的文件中
            line = json.dumps({
                'theme': adapter.get('theme'),
                'content': adapter.get('content'),
                'author': adapter.get('author', 'Unknown')  # 处理 author 字段，确保不存在时设为 'Unknown'
            }, ensure_ascii=False)

            self.files[theme].write(line + ',\n')
            spider.log(f"Writing item to file for theme: {theme}")
        except Exception as e:
            spider.log(f"Error writing item to file for theme {theme}: {e}")

        # MySQL 数据库插入部分
        try:
            self.cursor.execute('''
                INSERT INTO themes (theme, content, author) VALUES (%s, %s, %s)
            ''', (adapter.get('theme'), adapter.get('content'), adapter.get('author', 'Unknown')))
            self.conn.commit()
            spider.log(f"Data inserted into MySQL database for theme: {theme}")
        except mysql.connector.Error as err:
            spider.log(f"Error inserting data into MySQL for theme {theme}: {err}")
            self.conn.rollback()  # 发生错误时回滚

        return item

    def close_file(self, file, file_path):
        try:
            # 定位到文件末尾之前的两个字符（',\n'），并截取最后的逗号
            file.seek(file.tell() - 2, os.SEEK_SET)
            file.truncate()  # 删除最后的逗号
            file.write('\n]')  # 完成 JSON 数组
        except Exception as e:
            print(f"Error finishing JSON format in file {file_path}: {e}")
        finally:
            file.close()
