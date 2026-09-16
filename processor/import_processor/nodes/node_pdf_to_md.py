import json
from pathlib import Path
from plistlib import dumps

import logging

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")
        #检查和获取相关的参数
        #校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 获取上传链接并删除更换文件到mineru服务器
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        #下载zip压缩文件并解压改名
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        #读取md的内容
        with open(md_path, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()

        #设置state结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _step_1_validate_paths(self, state:ImportGraphState):
        #校验路径
        pdf_path = state.get("pdf_path")#这是上一个节点的,上一个节点传回给state了
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path",expected_type=str)
        file_dir = state.get("file_dir")#这里是导出文件的路径
        if not file_dir:
            raise StateFieldError(field_name="file_dir",expected_type=str)
        #封装路径,封装成对象
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)
        #判断文件是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{pdf_path}")
        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出的目录不存在:{output_dir_obj}")

        return  pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        logging.info(f"_step_2_upload_and_poll上传文件到mineru服务器...")
        return "下载url"

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        logging.info(f"_step_3_download_and_extract下载并解压改名")
        return "md_path"
#测试
if __name__ == "__main__":
    setup_logging()
    init_state={
        "pdf_path":r"D:\doc\H3C.pdf",
        "file_dir":r"D:\doc\output",
    }
    node = NodePDFToMD()
    result = node.process(init_state)#也可省略.proess,bsae.py文件中有默认执行
    json.dumps(result, ensure_ascii=False, indent=4)
    print(dumps)