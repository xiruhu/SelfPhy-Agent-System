"""
RAG 知识库管理器 (RAG Manager)
负责管理物理规则库，支持规则检索和验证
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class PhysicalRule:
    """物理规则"""
    rule_id: str
    rule_name: str
    logic_form: str
    natural_language: str
    verifier_code: Optional[str] = None
    failure_cases: List[str] = None
    priority: int = 5
    tags: List[str] = None

    def __post_init__(self):
        if self.failure_cases is None:
            self.failure_cases = []
        if self.tags is None:
            self.tags = []


class RAGManager:
    """RAG 知识库管理器"""

    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name="physical_rules",
            metadata={"description": "物理规则知识库"}
        )

        print(f"[RAG] Initialized with {self.collection.count()} rules")

    def add_rule(self, rule: PhysicalRule) -> None:
        """添加规则到数据库"""
        self.collection.add(
            ids=[rule.rule_id],
            documents=[rule.natural_language],
            metadatas=[{
                "rule_name": rule.rule_name,
                "logic_form": rule.logic_form,
                "priority": rule.priority,
                "tags": ",".join(rule.tags),
                "has_verifier": rule.verifier_code is not None
            }]
        )

        print(f"[RAG] Added rule: {rule.rule_id}")

    def retrieve_rules(
        self,
        query: str,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None
    ) -> List[PhysicalRule]:
        """
        检索相关规则

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            filter_tags: 标签过滤

        Returns:
            规则列表
        """
        where = None
        if filter_tags:
            # ChromaDB 的过滤语法
            where = {"tags": {"$contains": filter_tags[0]}}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where
        )

        rules = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                rule = PhysicalRule(
                    rule_id=results['ids'][0][i],
                    rule_name=metadata['rule_name'],
                    logic_form=metadata['logic_form'],
                    natural_language=results['documents'][0][i],
                    priority=metadata['priority'],
                    tags=metadata['tags'].split(',') if metadata['tags'] else []
                )
                rules.append(rule)

        return rules

    def verify_rule(
        self,
        rule: PhysicalRule,
        context: Dict[str, Any]
    ) -> bool:
        """
        执行规则验证

        Args:
            rule: 物理规则
            context: 验证上下文

        Returns:
            验证结果
        """
        if rule.verifier_code is None:
            return True

        # 简单的验证（实际应使用沙箱）
        try:
            # 创建安全的执行环境
            safe_globals = {
                "__builtins__": {},
                "context": context
            }

            # 执行验证代码
            exec(rule.verifier_code, safe_globals)
            return safe_globals.get("result", True)

        except Exception as e:
            print(f"[Warning] Rule verification failed: {e}")
            return False

    def initialize_default_rules(self) -> None:
        """初始化默认物理规则库"""
        default_rules = [
            PhysicalRule(
                rule_id="occlusion_persistence",
                rule_name="遮挡物体持久性",
                logic_form="occluded(O,t) ∧ ¬moved(O,t,t+k) → exists(O,pos,t+k)",
                natural_language="物体被遮挡后，如果没有外力作用，仍然存在于原位置。即使看不见，物体依然在那里。",
                priority=8,
                tags=["occlusion", "persistence", "physics"]
            ),
            PhysicalRule(
                rule_id="gravity_rule",
                rule_name="重力规则",
                logic_form="unsupported(O,t) → falls(O,t,t+k)",
                natural_language="物体失去支撑后会下落。悬浮的物体不符合物理定律（除非有特殊支撑）。",
                priority=9,
                tags=["gravity", "physics"]
            ),
            PhysicalRule(
                rule_id="spatial_transitivity",
                rule_name="空间传递性",
                logic_form="left_of(A,B) ∧ left_of(B,C) → left_of(A,C)",
                natural_language="如果A在B左边，B在C左边，则A在C左边。空间关系具有传递性。",
                priority=7,
                tags=["spatial", "reasoning", "transitivity"]
            ),
            PhysicalRule(
                rule_id="rotation_coordinate_transform",
                rule_name="旋转坐标变换",
                logic_form="rotate(agent, θ) → transform(left→front, front→right, right→back, back→left)",
                natural_language="当智能体旋转时，相对方向会改变。例如右转90度后，原来在左边的物体现在在前方。",
                priority=8,
                tags=["rotation", "coordinate", "spatial"]
            ),
            PhysicalRule(
                rule_id="distance_preservation",
                rule_name="距离保持性",
                logic_form="¬moved(O,t,t+k) ∧ moved(agent,d) → distance(agent,O,t+k) = distance(agent,O,t) ± d",
                natural_language="如果物体不动，智能体移动距离d，则智能体与物体的距离变化约为d（取决于移动方向）。",
                priority=7,
                tags=["distance", "spatial", "geometry"]
            ),
            PhysicalRule(
                rule_id="fov_constraint",
                rule_name="视场约束",
                logic_form="angle(agent_forward, object) > FOV/2 → ¬visible(object)",
                natural_language="物体在视场角之外时不可见。典型的人类视场角约为120度，超出此范围的物体看不见。",
                priority=6,
                tags=["fov", "visibility", "perception"]
            ),
            PhysicalRule(
                rule_id="depth_occlusion",
                rule_name="深度遮挡",
                logic_form="depth(A) > depth(B) ∧ overlap(A,B) → occluded(A,B)",
                natural_language="距离更远的物体会被距离更近的物体遮挡。深度信息决定遮挡关系。",
                priority=8,
                tags=["occlusion", "depth", "visibility"]
            ),
            PhysicalRule(
                rule_id="egocentric_reference_frame",
                rule_name="自我中心参考系",
                logic_form="egocentric(direction) → relative_to(agent_pose)",
                natural_language="第一人称视角下的方向（左、右、前、后）是相对于智能体当前朝向的，不是绝对方向。",
                priority=9,
                tags=["egocentric", "reference_frame", "spatial"]
            )
        ]

        # 检查是否已经初始化
        if self.collection.count() > 0:
            print("[RAG] Rules already initialized")
            return

        # 添加所有默认规则
        for rule in default_rules:
            self.add_rule(rule)

        print(f"[RAG] Initialized {len(default_rules)} default rules")

    def export_rules(self, output_path: str) -> None:
        """导出所有规则到 JSON 文件"""
        # 获取所有规则
        all_results = self.collection.get()

        rules = []
        for i in range(len(all_results['ids'])):
            metadata = all_results['metadatas'][i]
            rule = {
                "rule_id": all_results['ids'][i],
                "rule_name": metadata['rule_name'],
                "logic_form": metadata['logic_form'],
                "natural_language": all_results['documents'][i],
                "priority": metadata['priority'],
                "tags": metadata['tags'].split(',') if metadata['tags'] else []
            }
            rules.append(rule)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

        print(f"[RAG] Exported {len(rules)} rules to {output_path}")

    def import_rules(self, input_path: str) -> None:
        """从 JSON 文件导入规则"""
        with open(input_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)

        for rule_data in rules_data:
            rule = PhysicalRule(**rule_data)
            self.add_rule(rule)

        print(f"[RAG] Imported {len(rules_data)} rules from {input_path}")


def main():
    """主函数：初始化 RAG 知识库"""
    rag = RAGManager()

    # 初始化默认规则
    rag.initialize_default_rules()

    # 导出规则
    rag.export_rules("outputs/knowledge_base/physical_rules.json")

    # 测试检索
    print("\n[Test] Retrieving rules for 'occlusion'...")
    rules = rag.retrieve_rules("object is occluded but still exists", top_k=3)

    for rule in rules:
        print(f"\n- {rule.rule_name}")
        print(f"  Logic: {rule.logic_form}")
        print(f"  Description: {rule.natural_language}")
        print(f"  Priority: {rule.priority}")

    print(f"\n[Success] RAG system initialized with {rag.collection.count()} rules")


if __name__ == "__main__":
    main()
