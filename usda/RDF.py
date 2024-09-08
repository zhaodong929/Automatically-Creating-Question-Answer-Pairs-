from rdflib import Graph, URIRef, Literal, Namespace
from urllib.parse import quote

# 创建一个图
g = Graph()

# 定义命名空间，修改为与你的项目相关的命名空间
EX = Namespace("http://agriculture.org/")  # 使用与你的项目相关的命名空间

# 示例数据，假设从数据库读取
theme = "Animals"
sub_theme = "Nutrition security"
content = "What is nutrition security?"

# 对含空格的部分进行 URL 编码，确保其符合 URI 标准
encoded_sub_theme = quote(sub_theme)

# 创建 RDF 三元组并添加到图中
g.add((URIRef(EX[theme]), EX.hasSubTheme, Literal(sub_theme)))
g.add((URIRef(EX[encoded_sub_theme]), EX.hasContent, Literal(content)))

# 将图保存为 RDF 格式
g.serialize(destination='output.rdf', format='turtle')
