from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class RepeatingRowsFilter(BaseFilter):
    name = "RepeatingRows"
    _requires_dependencies = []

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        """    
        """
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        lines = doc.text.split("\n")
        # lines = [line for line in lines if line.strip() != ""]
        if len(lines) == 0:
            return False, "empty"
        unique_set = set()
        result = []
        for line in lines:  
            # 保留空行
            if line.strip() == "":
                result.append(line)
            else:
                # 如果是非空行，判断是否已经出现过
                if line not in unique_set:
                    unique_set.add(line)
                    result.append(line)
        
        doc.text = '\n'.join(result)        
        return True
