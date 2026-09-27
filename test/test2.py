from utils.embedding_utils import generate_embeddings, get_bge_m3_ef

model = get_bge_m3_ef()

# 测试1 直接用bge_m3客户端调用向量化方法，返回带有稠密和稀疏向量的
embeddings = model.encode_documents(["你好", "hello world"])
print(embeddings)

# embeddings = generate_embeddings(["你好", "hello world"])
#
# print(embeddings["dense"][0])
# print(embeddings["sparse"][0])