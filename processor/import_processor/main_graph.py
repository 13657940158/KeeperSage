import json
import logging

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    def __init__(self,config=None):
        self._compiled_graph = None#实例属性

    @property
    def graph(self):  # 可以通过KBImportWorkflow.graph使用，类方法
        """
        返回图实例
        """
        logging.info("获取图实例")
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()  # 如果图是空的，创建图
        return self._compiled_graph



    @staticmethod#工具方法
    def route_after_entry(state: ImportGraphState):
        # 这里由入口方法node_entry来告诉这个方法是pdf还是md
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            logging.info("route_after_entry路由器:未指定导入文件类型")
            return END



    def _build_graph(self):
        """
        创建主图
        """
        graph = StateGraph(ImportGraphState)
        # 注册节点
        graph.add_node("node_entry",NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img",NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition",NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 定义节点边
        graph.set_entry_point("node_entry")
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "node_md_img":"node_md_img",#需要一个方法来分辨走哪条分支，这里分别是方法的返回值和对应的返回节点
                "node_pdf_to_md":"node_pdf_to_md",
                END:END
            }
        )
        graph.add_edge("node_pdf_to_md","node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph_compile=graph.compile()
        return graph_compile
    #run函数
    def run(self,state: ImportGraphState,stream:bool=False):
        if stream:
            return self.graph.stream(state,stream_mode="values")
        else:
            return self.graph.invoke(state)
if __name__ == "__main__":
    # 启用日志
    setup_logging()
    workflow = KBImportWorkflow()
    init_state = {"import_file_path": r"D:\doc\H3C.md"}#文件路径
    for event in workflow.run(init_state,stream=True):
        print(f"state:{event}")
    # final_state = workflow.run(init_state,stream=False)
    # print(json.dumps(final_state,ensure_ascii=False,indent=4))
