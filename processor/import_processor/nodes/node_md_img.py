import logging

from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行")
        #1. 参数处理
        md_content,md_path_obj,images_dir = self._step_1_get_content(state)
        return state