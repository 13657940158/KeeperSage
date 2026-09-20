import json
import logging
import os.path
import re

from pathlib import Path
from typing import Tuple

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


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
        # #3. 视觉模型摘要
        # summaries = self._step_3_generate_summaries(md_path_obj.stem,target_images)#需要名字，所以要加stem
        #
        # #4.上传minio，替换md
        # new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)
        # # 5.备份
        # new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)
        #
        # state["md_content"] = new_md_content
        # state["md_path"] = new_md_file_name
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
            target_images.append((image_file,img_path,context))

        return target_images
    #步骤二方法1
    def _find_image_in_md(self, md_content: str, image_file: str,context_len:int = 100) -> Tuple[str,str]:
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match=pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0,start-context_len):start]
        post_text = md_content[end:min(len(md_content),end+context_len)]
        return pre_text,post_text



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