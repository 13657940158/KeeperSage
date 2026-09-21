import base64
import json
import logging
import os.path
import re
import time
from collections import deque

from pathlib import Path
from typing import Tuple, Dict, List, Deque

from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject

from config.lm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行")
        # 1. 参数处理
        md_content,md_path_obj,images_dir = self._step_1_get_content(state)
        print(f"md_content:{md_content},md_path_obj:{md_path_obj},images_dir:{images_dir}")
        #2.图片扫描
        target_images = self._step_2_scan_images(md_content,images_dir)
        print(f"")
        #3. 视觉模型摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem,target_images)#需要名字，所以要加stem

        #4.上传minio，替换md，把本地图片路径更改为minio服务器图片路径
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        # 5.备份
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state





    def _step_1_get_content(self,state):
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileExistsError(message=f"输入的文件不存在:{md_path}")
        md_content = state["md_content"]

        #测试共能代码，后期需要删除
        if not md_content:
            with open(md_path,"r",encoding="utf-8") as f:
                md_content = f.read()
        images_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self,md_content,images_dir):
       #1.返回结果

        target_images = []

        #2.扫描图片
        for image_file in os.listdir(images_dir):
            file_ext= os.path.splitext(image_file)[1].lower()#把文件名和扩展名分开，只取后面的扩展名，并且都转化为小写
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持，跳过:{image_file}")
                continue
            img_path = images_dir / image_file
            context=self._find_image_in_md(md_content,image_file)#寻找图片路径在md文档中的位置，找到图片的上下文
            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理:{image_file}")
                continue

            target_images.append((image_file,img_path,context))

        return target_images
    #步骤二的方法1
    def _find_image_in_md(self, md_content: str, image_file: str,context_len:int = 100) -> Tuple[str,str]:
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match=pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0,start-context_len):start]
        post_text = md_content[end:min(len(md_content),end+context_len)]
        return pre_text,post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
        str, str]:
        """
        步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """
        summaries = {}

        # 1、外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque = deque()

        # 2、循环处理图片
        for img_file, image_path, context in target_images:
            # 2.1、速率限制
            self._apply_api_rate_limit(request_deque, max_requests=10)
            #调用视觉模型的方法
            summaries[img_file] = self._summarize_image(image_path,root_folder=doc_stem,image_content=context)
        return summaries#返回图片摘要
    #步骤3的方法一,限制访问的次数
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")
    # 步骤3的方法二
    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        # 把图片转化成base64编码丢给大模型，大模型就看得懂了
        with open(image_path, "rb") as f:  # 必须用 "rb" 读取二进制文件
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        vl_ai = get_llm_client(lm_config.vl_model)  # 视觉模型的加载
        pre_text= ""
        post_text= ""
        if image_content:
            pre_text,post_text = image_content
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'这是"{root_folder}"文件中的一张图片，'
                            f'图片上文部分为"{image_content[0]}"，'
                            f'下文部分为"{image_content[1]}"，'
                            f'请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过15个字'
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]  # ⚠️ 原代码缺少这个闭合括号

        response = vl_ai.invoke(messages)
        return response.content.strip().replace("/n","")  # ⚠️ 原代码缺少返回值

    def _step_4_upload_and_replace(self, doc_stem:str, target_images:List[Tuple[str,str,Tuple[str,str]]], summaries:Dict[str,str], md_content:str):
        #0 加载minio客户端
        minio_client = get_minio_client()
        minio_img_dir = minio_config.img_dir#----------------------------------------------------------
        upload_dir=f"{minio_img_dir}/{doc_stem}"#分别是project/file项目名
        print(f"将图片存入{upload_dir}目录")

        #1 清理,minio目录
        self.clean_minio_directory(minio_client, upload_dir)

        #2 上传,批量图片，返回上传之后的url
        urls=self.upload_images_batch(minio_client,upload_dir,target_images)

        #3 合并,将摘要和url路径
        image_info = self.merge_summary_and_url(summaries,urls)

        #4 替换,md文件的摘要和路径
        md_content = self.process_md_file(md_content,image_info)#替换摘要和路径

        return md_content
    #步骤四方法一，非必要
    def clean_minio_directory(self, minio_client: Minio,upload_dir) -> None:
        """
        幂等性清理：上传前先删除 MinIO 中指定目录下的旧文件。
        防止重名文件导致的内容混淆或垃圾堆积。
        :param minio_client: 初始化完成的MinIO客户端对象
        :param prefix: MinIO目录前缀（要清理的目录路径）
        """
        try:
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)
            # 构造删除列表
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                for error in errors:
                    self.logger.error(f"删除失败：{error}")
        except Exception as e:
            self.logger.error(f"清理minio目录失败：{e}")

    # Dict[str,Tuple[str,str]]) ——> 顾名思义，就是字典中 ：图片名，（图片url，图片概括）
    # 步骤四方法二，上传文件
    def upload_images_batch(self, minio_client:Minio, upload_dir:str, target_images:List[Tuple[str,str,Tuple[str,str]]]):
        urls={}
        for img_file,img_path,_ in target_images:
            #上传
            object_name = f"{upload_dir}/{img_file}"#minio的文件对象，img_file已被处理，有后缀名，这是个路径
            print(f"上传文件:{object_name}")
            urls[img_file] = self.upload_to_minio(minio_client,img_path,object_name)#返回minio的url

        return urls
    # 步骤四方法三，合并参数
    def merge_summary_and_url(self, summaries:Dict[str,str], urls)->Dict[str,Tuple[str,str]]:
        image_info = {}
        for image_file,summary in summaries.items():
            image_info[image_file] = (summary,urls[image_file])
        return image_info
    # 步骤四方法四，替换md中的url和摘要
    def process_md_file(self, md_content:str, image_info:Dict[str,Tuple[str,str]]):#Dict[str,Tuple[str,str]]---->图片名称,[图片摘要,url]
        for image_file,(summary,url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(f"![{summary}]({url})", md_content)#替换代码核心
        return md_content
    #步骤四的方法五
    def upload_to_minio(self, minio_client:Minio, img_path:str, object_name:str)->str:
        #上传minio
        ifSuccess = minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=img_path,
            content_type=f"image/{os.path.splitext(img_path)[1][1:]}"

        )
        endpoint = minio_config.endpoint
        url = f"http://{endpoint}/{minio_config.bucket_name}/{object_name}"
        return url
    #步骤五 备份新文档
    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        # 构造新文件路径：替换原后缀为 _new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")

        return new_md_file_name
if __name__ == "__main__":
    setup_logging()

    init_state = {
        "md_path": r"D:\doc\output\H3C\H3C.md",
        "md_content": None
    }
    node = NodeMDImg()
    result = node(init_state)

    dumps = json.dumps(result)
    print(dumps)