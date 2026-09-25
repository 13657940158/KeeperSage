
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState

from typing import Tuple, List, Dict
class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        #1.参数处理
        content,file_title = self._step_1_get_inputs(state)
        #2.标题切（初切）
        sections,title_count,lines_count=self._step_2_split_by_title(content,file_title)#需要用字典列表

        #3.无标题兜底（默认标题）
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)
        # 4.快精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)
        for section in sections:
            print(f"{section['content']}")
            print("============================================================================")
        print(f"{title_count}")
        print(f"{lines_count}")
        #5.打印日志
        self._step_5_print_stats(lines_count, sections)
        # 6.备份
        self._step_6_backup(state, sections)
        state["chunks"] = None
        return state

    def _step_1_get_inputs(self, state):
        print("node_document_split:步骤1：参数处理")
        content = state.get('md_content')
        file_title = state.get('file_title')
        if not content:
            raise StateFieldError(field_name="file_title",message="文件标题不能为空",expected_type=str)
        #标准化处理
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return content,file_title


    def _step_2_split_by_title(self, content, file_title):
        print("node_document_split:步骤2：标题初切")
        #参数声明
        sections: List[Dict[str,str]]= []
        title_count:int=0
        lines = content.split('\n')
        current_lines = []
        #切块逻辑（标题切）
        in_code_block = False#判断是否在md文档中的代码块里面
        title_pattern = r'\s*#{1,6}\s+.+'#标题正则
        current_title = ""

        def _flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),
                "parent_title": "",
                "file_title": file_title,
            })

        for line in lines:
            striped_line=line.strip()
            #判断是否在代码块中
            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
            if (not in_code_block) and (re.match(title_pattern,line)):
                # 如果不在代码块里，并且这一行是标题：
                #   说明换标题了，把上一个标题和 current_lines 打包成 section
                #   然后当前行变成新标题，清空 current_lines
                #
                # 否则：
                #   包括两种情况：
                #   1. 在代码块里，不管是不是标题，都当普通行
                #   2. 不在代码块里，但不是标题，也当普通行
                #   这两种都直接加到 current_lines
                _flush_section()#这里把之前标题全部打包
                current_title=striped_line#这里已经开始换一个新的标题
                current_lines = [current_title]#继续，这里要初始化设置为空，把上一个setions的内容清空
                title_count+=1
            else:
                current_lines.append(line)#普通行或者代码块
        _flush_section()
        return sections,title_count,len(lines)

    #步骤3无标题兜底
    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split:步骤3：无标题兜底（默认标题）")
        if title_count==0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections#无标题只是帮忙返回一个标题
    #步骤4精细化处理（长切短合）
    def _step_4_refine_chunks(self, sections):
        print("node_document_split:步骤4：块精细化处理（长切短合）")
        #先长切，在短合，比较适合
        #长切列表
        refined_split = []
        for sec in sections:
            refined_split.extend(self.split_long_section(sec))#长切操作

            #短合列表
            #合并前提，1.足够短，2.属于同一父标题
        final_sections = self.merge_short_sections(refined_split)#短合操作

        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections


    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split:步骤5：打印日志")
        pass

    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6：备份")
        pass
    #切分方法长
    def split_long_section(self, section:Dict[str,str]):
        print("node_document_split:步骤4方法1长切")
        content = section.get("content","")
        content_len = len(content)
        #长度合标，直接返回
        if content_len <=get_config().max_content_length:
            return [section]
        title = section.get("title")
        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - len(prefix)

        #去标题
        body = content
        if title and body.lstrip().startswith(title):#查找body的开头有没有重复的标题
            body= body[body.find(title) + len(title):].lstrip()

        #切分
        splitter =RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )
        #切分结果
        sub_sections = []

        for index , chunk in enumerate(splitter.split(body),start=1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix + text).strip()
            sub_sections.append({
                "title": f"{title}-{index}" if title else f"chunk-{index}",  # 子Chunk标题（带序号）
                "content": full_text,  # 切分后的完整内容
                "parent_title": title,  # 父章节标题（用于后续合并）
                "part": index,  # 子Chunk序号
                "file_title": section.get("file_title"),  # 所属文件标题
            })

        return sub_sections
    #短合
    def merge_short_sections(self, refined_split):
        print("node_document_split:步骤4方法2短合")
        return refined_split


if __name__ == '__main__':
    node = NodeDocumentSplit()

    with open("D:\doc\output\H3C\H3C_new.md","r",encoding="utf-8") as f:
        md_content = f.read()

        init_state = {
            "md_path":"D:\doc\output\H3C\H3C_new.md",
            "md_content":md_content,
            "file_title":"H3C_new",
        }

        process = node.process(init_state)
        print(f"切分节点执行流程：{process}")





