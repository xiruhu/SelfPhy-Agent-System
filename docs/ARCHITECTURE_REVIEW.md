# SelfPhy-Agent-System 架构审查与增强方案

**审查时间**: 2026-06-04  
**审查人**: 首席技术顾问（Claude Sonnet 4.6）  
**项目现状**: 已有基础框架，需要系统性增强与学术化升级

---

## 📋 执行摘要

已成功完成对 SelfPhy-Agent-System 的深度架构审查。系统基础框架完整，但存在 5 个关键技术坑点需要升级。本方案提供了详细的改进路线图，预计 8 天完成全部增强。

**核心发现**：
- ✅ 模块分层清晰，非对称式长官架构设计合理
- 🔴 数据表征不够语义化（四元数 vs 转身90度）
- 🔴 RAG 缺乏可执行验证（纯文本 vs 工具调用）
- 🔴 诊断缺少证据链（规则匹配 vs 数值证据）
- 🔴 数据流缺乏类型安全（Dict vs Pydantic）
- 🔴 VL-LN-Bench 数据集尚未适配

---

## 一、现有架构优势

### ✅ 做得好的地方

1. **清晰的模块分层**：数据感知 → 出题 → 测试 → 诊断 → 可视化，流程完整
2. **非对称式长官架构**：Claude Sonnet 4.6 作为 Supervisor 的设计合理
3. **统一适配器接口**：evaluate_runner.py 支持多模型派发
4. **RAG 知识库基础**：ChromaDB + 8 条物理规则，有扩展空间
5. **完整的数据链路**：从 Habitat 到最终报告的输出目录结构清晰

---

## 二、核心架构问题诊断

### 🔴 问题 1：Habitat 位移数据的表征方式不够语义化

**改进方案**：语义化 + 增量描述 + 保留原始数据  
**实施位置**：core/cv2_processor.py → 新增 TrajectorySemanticizer 类

### 🔴 问题 2：RAG 知识库的白盒化程度不足

**改进方案**：三级索引体系
- Level 1: 物理定律层（可证伪）+ 工具触发器
- Level 2: 空间几何工具链（可计算）
- Level 3: 案例记忆层（可检索）

**实施位置**：core/rag_manager.py → 重构为 EnhancedRAGManager

### 🔴 问题 3：四步排除法缺乏可解释性

**改进方案**：5D 诊断矩阵
1. 物理-语义双向对齐 (Bidirectional Grounding)
2. 时序空间记忆重建 (Mental Map Reconstruction)
3. 多模态视场反事实验证 (Counterfactual FOV Check)
4. 根因分类的可解释决策树 (Evidence Chain)
5. 对抗性案例自动生成 (Adversarial Case Mining)

**实施位置**：core/claude_reflector.py → 升级为 EnhancedInspectorAgent

### 🔴 问题 4：数据流缺乏类型安全

**改进方案**：Pydantic 统一数据模型 + 血缘元数据追踪  
**实施位置**：新建 schema/ 目录

### 🔴 问题 5：VL-LN-Bench 数据集尚未适配

**改进方案**：实现 Parquet + trajectory.json 解析器  
**实施位置**：core/habitat_loader.py 完整实现

---

## 三、实施优先级与时间规划

### Phase 1: 基础设施强化（2 天）
1. 创建 Pydantic 数据模型（schema/core_types.py）
2. 重构 RAG 为三级架构（core/rag_manager_enhanced.py）
3. 实现 VL-LN-Bench 加载器（core/habitat_loader.py）

### Phase 2: 诊断系统升级（3 天）
4. 实现 5D 诊断矩阵（core/inspector_enhanced.py）
5. 添加空间几何工具链（tools/spatial_geometry.py）
6. 实现 Evidence Chain 生成

### Phase 3: 可视化与论文素材（2 天）
7. Mental Map 热力图
8. 对抗性案例生成器
9. 跨模型对比雷达图

### Phase 4: 端到端集成（1 天）
10. 更新主流水线（run_pipeline_enhanced.py）
11. 批量测试 D7N2EKCX4Sj 数据集
12. 生成论文级报告与图表

---

## 四、立即行动建议

建议按以下顺序展开：

1. **Pydantic 数据模型**（基础设施，影响所有模块）
2. **VL-LN-Bench 加载器**（解锁真实数据测试）
3. **增强版 RAG**（支撑诊断系统升级）
4. **5D 诊断矩阵**（核心学术贡献）

---

## 五、预期成果

### 学术贡献
1. 新型评测范式：第一人称动态物理世界的系统化评测框架
2. 可解释 AI 诊断：Evidence Chain + Mental Map 的白盒化错误分析
3. 对抗性测试闭环：从错误中自动生成针对性测试案例

### 工程价值
1. 类型安全的数据流：Pydantic 模型 + 血缘追踪
2. 模块化可扩展架构：接口抽象 + 事件驱动
3. 多模型统一评测平台：一键对比 Kimi/MiniMax/豆包

---

**完整技术细节请查看项目文档目录下的详细设计文档**

文档创建时间: 2026-06-04  
下次审查: 完成 Phase 1 后（预计 2 天后）
