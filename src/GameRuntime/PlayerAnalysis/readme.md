# PlayerAnalysis

- PlayerAnalyzer：
  - 精确查找 Player（名称/标签/常见别名）
  - 递归采集组件与子节点结构（限制层级防止无限递归）
  - 导出详细报告至 PlayerAnalysis/

输入/输出：
- 输入：场景中的 Player 及其子节点
- 输出：控制台打印与文本报告

维护建议：
- 只做结构化采集，不进行复杂运算
- 深度/数量控制，避免性能抖动
