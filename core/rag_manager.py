"""
rag_manager_v2.py
-----------------
增强版 RAG 知识库管理器

核心改进：
1. 三层知识体系：空间几何定理 + 物理常识 + 视觉遮挡规则
2. 结构化规则存储（规则ID、类别、优先级、验证器）
3. 语义检索 + 标签过滤 + 优先级排序
4. 规则链推理（多规则组合验证）
5. 可视化规则关系图

数据结构：
  Rule:
    - rule_id: 唯一标识
    - category: spatial_geometry / physical_law / visual_occlusion
    - rule_name: 规则名称
    - logic_form: 形式化逻辑表达
    - natural_language: 自然语言描述
    - verifier_code: Python 验证代码
    - examples: 正例和反例
    - priority: 优先级 (1-10)
    - dependencies: 依赖的其他规则
    - tags: 标签列表
"""

import json
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class RuleCategory(str, Enum):
    """规则类别"""
    SPATIAL_GEOMETRY = "spatial_geometry"  # 空间几何定理
    PHYSICAL_LAW = "physical_law"  # 物理常识
    VISUAL_OCCLUSION = "visual_occlusion"  # 视觉遮挡规则


@dataclass
class RuleExample:
    """规则示例"""
    description: str
    is_positive: bool  # True=正例，False=反例
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """知识规则"""
    rule_id: str
    category: RuleCategory
    rule_name: str
    logic_form: str
    natural_language: str
    verifier_code: Optional[str] = None
    examples: List[RuleExample] = field(default_factory=list)
    priority: int = 5  # 1-10, 数字越大优先级越高
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他规则ID
    tags: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict:
        """转换为字典"""
        d = asdict(self)
        d['category'] = self.category.value
        return d


class RAGManagerV2:
    """增强版 RAG 知识库管理器"""

    def __init__(self, db_path: str = "./data/chroma_db_v2"):
        """
        初始化 RAG 管理器

        Args:
            db_path: ChromaDB 数据库路径
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # 为每个类别创建独立的集合
        self.collections = {
            RuleCategory.SPATIAL_GEOMETRY: self.client.get_or_create_collection(
                name="spatial_geometry_rules",
                metadata={"description": "空间几何定理"}
            ),
            RuleCategory.PHYSICAL_LAW: self.client.get_or_create_collection(
                name="physical_law_rules",
                metadata={"description": "物理常识规则"}
            ),
            RuleCategory.VISUAL_OCCLUSION: self.client.get_or_create_collection(
                name="visual_occlusion_rules",
                metadata={"description": "视觉遮挡规则"}
            )
        }

        self._print_status()

    def _print_status(self):
        """打印初始化状态"""
        print(f"\n[RAG V2] 知识库初始化完成")
        for category, collection in self.collections.items():
            count = collection.count()
            print(f"  - {category.value}: {count} 条规则")

    def add_rule(self, rule: Rule) -> None:
        """
        添加规则到知识库

        Args:
            rule: Rule 对象
        """
        collection = self.collections[rule.category]

        # 准备元数据
        metadata = {
            "rule_name": rule.rule_name,
            "logic_form": rule.logic_form,
            "priority": rule.priority,
            "tags": ",".join(rule.tags),
            "has_verifier": rule.verifier_code is not None,
            "has_examples": len(rule.examples) > 0,
            "dependencies": ",".join(rule.dependencies)
        }

        # 添加到 ChromaDB
        collection.upsert(
            ids=[rule.rule_id],
            documents=[rule.natural_language],
            metadatas=[metadata]
        )

        # 保存完整规则到 JSON（用于恢复详细信息）
        rules_dir = self.db_path / "rules_json"
        rules_dir.mkdir(exist_ok=True)

        rule_file = rules_dir / f"{rule.rule_id}.json"
        with open(rule_file, 'w', encoding='utf-8') as f:
            json.dump(rule.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[RAG V2] 添加规则: {rule.rule_id} ({rule.category.value})")

    def retrieve_rules(
        self,
        query: str,
        categories: Optional[List[RuleCategory]] = None,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None,
        min_priority: int = 0
    ) -> List[Rule]:
        """
        检索相关规则

        Args:
            query: 查询文本
            categories: 要搜索的类别列表（None=所有类别）
            top_k: 每个类别返回的最大规则数
            filter_tags: 标签过滤
            min_priority: 最小优先级

        Returns:
            规则列表（按优先级和相关性排序）
        """
        if categories is None:
            categories = list(RuleCategory)

        all_rules = []

        for category in categories:
            collection = self.collections[category]

            if collection.count() == 0:
                continue

            # 构建过滤条件
            where = {}
            if min_priority > 0:
                where["priority"] = {"$gte": min_priority}

            # 执行检索
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, collection.count()),
                    where=where if where else None
                )

                # 重建 Rule 对象
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        rule_id = results['ids'][0][i]
                        metadata = results['metadatas'][0][i]

                        # 从 JSON 文件恢复完整规则
                        rule = self._load_rule_from_json(rule_id)

                        if rule:
                            # 标签过滤
                            if filter_tags and not any(tag in rule.tags for tag in filter_tags):
                                continue

                            all_rules.append(rule)

            except Exception as e:
                print(f"[Warning] 检索 {category.value} 时出错: {e}")

        # 按优先级排序
        all_rules.sort(key=lambda r: r.priority, reverse=True)

        return all_rules[:top_k * len(categories)]

    def _load_rule_from_json(self, rule_id: str) -> Optional[Rule]:
        """从 JSON 文件加载完整规则"""
        rule_file = self.db_path / "rules_json" / f"{rule_id}.json"

        if not rule_file.exists():
            return None

        try:
            with open(rule_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重建 RuleExample 对象
            examples = [RuleExample(**ex) for ex in data.get('examples', [])]

            # 重建 Rule 对象
            return Rule(
                rule_id=data['rule_id'],
                category=RuleCategory(data['category']),
                rule_name=data['rule_name'],
                logic_form=data['logic_form'],
                natural_language=data['natural_language'],
                verifier_code=data.get('verifier_code'),
                examples=examples,
                priority=data.get('priority', 5),
                dependencies=data.get('dependencies', []),
                tags=data.get('tags', []),
                created_at=data.get('created_at', '')
            )

        except Exception as e:
            print(f"[Warning] 加载规则 {rule_id} 失败: {e}")
            return None

    def verify_with_rule(
        self,
        rule: Rule,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        使用规则验证上下文

        Args:
            rule: 规则对象
            context: 验证上下文

        Returns:
            (是否通过, 详细信息)
        """
        if rule.verifier_code is None:
            return True, "无验证器，默认通过"

        try:
            # 创建安全的执行环境
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "round": round,
                    "len": len
                },
                "context": context
            }

            # 执行验证代码
            exec(rule.verifier_code, safe_globals)
            result = safe_globals.get("result", True)
            message = safe_globals.get("message", "")

            return result, message

        except Exception as e:
            return False, f"验证器执行失败: {str(e)}"

    def verify_with_chain(
        self,
        rule_ids: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        规则链验证（多个规则组合验证）

        Args:
            rule_ids: 规则ID列表
            context: 验证上下文

        Returns:
            验证结果字典
        """
        results = {
            "all_passed": True,
            "rule_results": []
        }

        for rule_id in rule_ids:
            rule = self._load_rule_from_json(rule_id)

            if rule is None:
                results["rule_results"].append({
                    "rule_id": rule_id,
                    "passed": False,
                    "message": "规则不存在"
                })
                results["all_passed"] = False
                continue

            passed, message = self.verify_with_rule(rule, context)

            results["rule_results"].append({
                "rule_id": rule_id,
                "rule_name": rule.rule_name,
                "passed": passed,
                "message": message
            })

            if not passed:
                results["all_passed"] = False

        return results

    def initialize_default_rules(self):
        """初始化默认规则库"""
        print("\n[RAG V2] 初始化默认规则库...")

        # === 1. 空间几何定理 ===
        spatial_rules = [
            Rule(
                rule_id="rotation_coordinate_transform",
                category=RuleCategory.SPATIAL_GEOMETRY,
                rule_name="旋转坐标变换定理",
                logic_form="rotate(agent, θ) → transform_coordinates(relative_positions, θ)",
                natural_language="智能体旋转时，所有物体的相对方向按旋转角度变换。右转90°：左→前，前→右，右→后，后→左。",
                priority=9,
                tags=["rotation", "coordinate", "transformation"],
                examples=[
                    RuleExample(
                        description="右转90度后，原左侧物体在前方",
                        is_positive=True,
                        context={"rotation": 90, "original_direction": "left", "new_direction": "front"}
                    )
                ]
            ),
            Rule(
                rule_id="spatial_transitivity",
                category=RuleCategory.SPATIAL_GEOMETRY,
                rule_name="空间关系传递性",
                logic_form="left_of(A,B) ∧ left_of(B,C) → left_of(A,C)",
                natural_language="如果A在B左边，B在C左边，则A在C左边。空间关系具有传递性。",
                priority=7,
                tags=["transitivity", "spatial", "reasoning"]
            ),
            Rule(
                rule_id="distance_triangle_inequality",
                category=RuleCategory.SPATIAL_GEOMETRY,
                rule_name="三角不等式",
                logic_form="distance(A,C) ≤ distance(A,B) + distance(B,C)",
                natural_language="两点间直线距离最短。绕路的距离一定大于等于直线距离。",
                priority=6,
                tags=["distance", "geometry", "inequality"]
            ),
            Rule(
                rule_id="euclidean_distance_formula",
                category=RuleCategory.SPATIAL_GEOMETRY,
                rule_name="欧式距离公式",
                logic_form="distance(p1, p2) = √((x2-x1)² + (y2-y1)² + (z2-z1)²)",
                natural_language="三维空间中两点的距离由坐标差计算。",
                priority=8,
                tags=["distance", "calculation", "geometry"],
                verifier_code="""
import math
p1 = context.get('p1', [0,0,0])
p2 = context.get('p2', [0,0,0])
expected = context.get('expected_distance', 0)
calculated = math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))
result = abs(calculated - expected) < 0.1
message = f"计算距离: {calculated:.2f}, 期望: {expected:.2f}"
"""
            )
        ]

        # === 2. 物理常识规则 ===
        physical_rules = [
            Rule(
                rule_id="object_persistence",
                category=RuleCategory.PHYSICAL_LAW,
                rule_name="物体持久性原理",
                logic_form="exists(O, t) ∧ ¬destroyed(O, t, t+k) → exists(O, t+k)",
                natural_language="物体不会凭空消失。如果物体存在且没被销毁，它会一直存在。",
                priority=9,
                tags=["persistence", "object", "physics"]
            ),
            Rule(
                rule_id="gravity_law",
                category=RuleCategory.PHYSICAL_LAW,
                rule_name="重力定律",
                logic_form="unsupported(O) → falls(O)",
                natural_language="无支撑的物体会下落。悬浮物体违反物理定律（除非有隐藏支撑）。",
                priority=8,
                tags=["gravity", "physics", "support"]
            ),
            Rule(
                rule_id="static_object_immobility",
                category=RuleCategory.PHYSICAL_LAW,
                rule_name="静态物体不动性",
                logic_form="static(O) ∧ ¬force_applied(O) → position(O, t) = position(O, t+k)",
                natural_language="静态物体（家具、墙壁等）不会自己移动。没有外力作用时位置保持不变。",
                priority=8,
                tags=["static", "immobility", "physics"]
            )
        ]

        # === 3. 视觉遮挡规则 ===
        occlusion_rules = [
            Rule(
                rule_id="occlusion_persistence",
                category=RuleCategory.VISUAL_OCCLUSION,
                rule_name="遮挡后物体持久性",
                logic_form="occluded(O, t) ∧ static(O) → exists(O, same_position, t+k)",
                natural_language="物体被遮挡后仍存在于原位置。看不见不等于不存在。",
                priority=9,
                tags=["occlusion", "persistence", "memory"],
                examples=[
                    RuleExample(
                        description="桌子被墙遮挡后，转回来仍在原位",
                        is_positive=True,
                        context={"object": "table", "occluded_at": 10, "checked_at": 50}
                    )
                ]
            ),
            Rule(
                rule_id="fov_constraint",
                category=RuleCategory.VISUAL_OCCLUSION,
                rule_name="视场角约束",
                logic_form="angle(forward, object) > FOV/2 → ¬visible(object)",
                natural_language="超出视场角的物体不可见。人类水平视场约120°，垂直约90°。",
                priority=7,
                tags=["fov", "visibility", "angle"]
            ),
            Rule(
                rule_id="depth_occlusion",
                category=RuleCategory.VISUAL_OCCLUSION,
                rule_name="深度遮挡规则",
                logic_form="depth(A) < depth(B) ∧ overlaps(A, B) → occludes(A, B)",
                natural_language="近处物体遮挡远处物体。前景遮挡背景。",
                priority=8,
                tags=["depth", "occlusion", "layering"]
            ),
            Rule(
                rule_id="partial_visibility",
                category=RuleCategory.VISUAL_OCCLUSION,
                rule_name="部分可见性",
                logic_form="partially_visible(O) → exists(O) ∧ partially_occluded(O)",
                natural_language="部分可见的物体完整存在，只是部分被遮挡。看到物体一角可以推断整体存在。",
                priority=6,
                tags=["partial", "visibility", "inference"]
            )
        ]

        # 批量添加规则
        for rule in spatial_rules + physical_rules + occlusion_rules:
            self.add_rule(rule)

        print(f"[RAG V2] ✓ 默认规则库初始化完成")
        print(f"  - 空间几何: {len(spatial_rules)} 条")
        print(f"  - 物理常识: {len(physical_rules)} 条")
        print(f"  - 视觉遮挡: {len(occlusion_rules)} 条")

    def export_rules(self, output_path: str):
        """导出所有规则到 JSON 文件"""
        rules_json_dir = self.db_path / "rules_json"
        all_rules = []

        for rule_file in rules_json_dir.glob("*.json"):
            with open(rule_file, 'r', encoding='utf-8') as f:
                all_rules.append(json.load(f))

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_rules, f, indent=2, ensure_ascii=False)

        print(f"[RAG V2] 导出 {len(all_rules)} 条规则到: {output_path}")


def main():
    """测试脚本"""
    print("=" * 60)
    print("RAG Manager V2 - 测试脚本")
    print("=" * 60)

    # 初始化管理器
    rag = RAGManagerV2()

    # 初始化默认规则
    rag.initialize_default_rules()

    # 测试检索
    print("\n" + "=" * 60)
    print("[测试] 检索与旋转相关的规则")
    print("=" * 60)

    rules = rag.retrieve_rules(
        query="智能体旋转后物体方向如何变化？",
        top_k=3
    )

    for i, rule in enumerate(rules, 1):
        print(f"\n{i}. {rule.rule_name}")
        print(f"   类别: {rule.category.value}")
        print(f"   优先级: {rule.priority}")
        print(f"   描述: {rule.natural_language}")

    # 导出规则
    print("\n" + "=" * 60)
    print("[导出] 保存规则到文件")
    print("=" * 60)
    rag.export_rules("outputs/rules_export_v2.json")


if __name__ == "__main__":
    main()
