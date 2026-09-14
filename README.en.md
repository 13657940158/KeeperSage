"""
    工程环境
"""
conda create -n py311 python=3.11
conda activate py311
python -m venv kb311
.\Scripts\activate

"""
    pytorch
"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 (5090)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 (5090)
cuda开发工具：https://developer.nvidia.com/cuda-12-4-0-download-archive
检查pytorch状态：python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

测试：
import torch
print(torch.cuda.is_available())  # 如果输出 True，说明成功认出显卡！
print(torch.version.cuda)         # 这会显示你安装的 PyTorch 自带的 CUDA 版本（比如 12.1）

"""
    mineru
"""
官方：https://opendatalab.github.io/MinerU/zh/
官方：https://mineru.net/apiManage/docs
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
测试： mineru --version

"""
    mineru模型
"""
模型缓存：$env:MODELSCOPE_CACHE="D:\ai_models\modelscope_cache"
下模型：mineru-models-download
桌面客户端：https://mineru.net/
测试：mineru -p xxx.pdf -o output --backend pipeline

"""
    milvus
"""
pip uninstall -y pymilvus milvus-model FlagEmbedding
uv add pymilvus[model] transformers FlagEmbedding
pip install "pymilvus[model]" transformers FlagEmbedding

"""
    安装BGE-M3
"""
uv add pymilvus[model] transformers FlagEmbedding
pip install "pymilvus[model]" transformers FlagEmbedding
pip install modelscope
执行脚本：python download_bgem3.py
"""
    from modelscope.hub.snapshot_download import snapshot_download
    # 下载模型到当前目录下的 models/bge-m3 文件夹
    model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
    print(f"模型已下载到: {model_dir}")
"""
测试：python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
万一测试不可用：pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 (仅限于5090)



"""
    fastapi
"""
pip install fastapi
pip install "uvicorn[standard]"



"""
    清华镜像
"""
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

"""
    若安装minio会报错，升级版本
"""
python -m pip install --upgrade pip


knowledge/
├── api/                              # API 路由层
│   ├── query_router.py              # 查询服务路由 (port 8001)
│   │   ├── POST /query              # 发起查询
│   │   ├── GET /stream/{session_id} # SSE 流式获取
│   │   ├── GET /history/{session_id}# 获取历史
│   │   └── DELETE /history/...      # 清除历史
│   └── import_router.py             # 导入服务路由 (port 8000)
│       ├── POST /upload             # 上传文件
│       └── GET /status/{task_id}    # 查询任务状态
│
├── core/                             # 核心配置
│   ├── deps.py                      # 依赖注入（单例管理）
│   └── paths.py                     # 路径常量配置
│
├── processor/                        # 业务处理流程（LangGraph）
│   ├── import_processor/               # 导入流程
│   │   ├── base.py           		 # 导入节点基类
│   │   ├── config.py           	  # 导入流程配置管理
│   │   ├── exceptions.py             # 导入流程自定义异常
│   │   ├── main_graph.py           # 导入流程图定义
│   │   ├── state.py                # 状态类型定义
│   │   └── nodes/                  # 处理节点
│   │       ├── node_entry.py            # 入口节点
│   │       ├── node_pdf_to_md.py        # PDF 转 MD
│   │       ├── node_md_img.py           # 图片处理
│   │       ├── node_document_split.py   # 文档切分
│   │       ├── node_item_name_recognition.py  # 商品识别
│   │       ├── node_bge_embedding.py    # 向量嵌入
│   │       └── node_import_milvus.py    # Milvus 存储
│   │
│   └── query_processor/              	# 查询流程
│       ├── base.py           		    # 查询节点基类
│       ├── config.py           	    # 查询流程配置管理
│       ├── exceptions.py               # 查询流程自定义异常
│       ├── main_graph.py           	# 查询流程图定义
│       ├── state.py                	# 状态类型定义
│       ├── prompt.py               	# 提示词模板
│       └── nodes/                  	# 处理节点
│           ├── node_item_name_confirm.py    # 商品确认
│           ├── node_vector_search.py        # 向量检索
│           ├── node_hyde_search.py          # HyDE 检索
│           ├── node_web_search_mcp.py       # Web 搜索
│           ├── node_rrf.py                  # RRF 融合
│           ├── node_rerank.py               # 重排序
│           └── node_answer_output.py        # 答案生成
│
├── schema/                          # 数据模型定义
│   ├── query_schema.py              # 查询请求/响应模型
│   ├── upload_schema.py             # 上传响应模型
│   └── task_schema.py               # 任务状态模型
│
├── services/                        # 业务服务层
│   ├── file_import_service.py       # 文件导入服务
│   └── task_service.py              # 任务管理服务
│
├── utils/                           # 工具函数库
│   ├── milvus_utils.py              # Milvus 向量库操作
│   ├── embedding_utils.py           # 向量嵌入工具
│   ├── llm_utils.py                 # LLM 客户端封装
│   ├── reranker_utils.py            # Reranker 模型
│   ├── mongo_history_utils.py       # MongoDB 历史记录
│   ├── sse_utils.py                 # SSE 流式推送
│   ├── task_utils.py                # 任务状态管理
│   └── minio_utils.py               # MinIO 对象存储
│
├── front/                           # 前端页面
│   ├── chat.html                    # 聊天界面
│   └── import.html                  # 导入界面
│
├── test/                             # 测试代码
├── docs/                             # 文档目录
├── temp_data/                        # 临时数据目录
├── .env                              # 环境配置文件
├── .env.example                      # 环境配置文件模板
├── pyproject.toml                    # Python 依赖声明
└── uv.locl							# uv版本锁定文件

"""