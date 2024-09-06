# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class UsdaSpiderItem(scrapy.Item):
    theme = scrapy.Field()
    sub_theme = scrapy.Field()
    content = scrapy.Field()
    link = scrapy.Field()
    level = scrapy.Field()
    parent_link = scrapy.Field()
    title = scrapy.Field()  # 添加title字段
    author = scrapy.Field()  # 添加author字段
    publish_date = scrapy.Field()  # 添加publish_date字段
