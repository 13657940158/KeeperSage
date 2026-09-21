import base64

from config.lm_config import lm_config
from utils.llm_utils import get_llm_client

bsae_str=None
    # 把图片转化成base64编码丢给大模型，大模型就看得懂了
with open(r"C:\Users\24764\OneDrive\图片\【哲风壁纸】卡通羊-可爱小羊.jpg", "rb") as f:  # 必须用 "rb" 读取二进制文件
    bsae_str = base64.b64encode(f.read()).decode("utf-8")

vl_ai = get_llm_client(lm_config.vl_model)  # 视觉模型的加载

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f'这是书籍文件中的一张图片，'
                    f'图片上文部分为"这是一个实验图"，'
                    f'下文部分为"是测试的"，'
                    f'请用中文简要总结这张图片的内容，用于 Markdown 图片标题。'
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{bsae_str}"
                }
            }
        ]
    }
]  #  原代码缺少这个闭合括号

response = vl_ai.invoke(messages)
print(response)
