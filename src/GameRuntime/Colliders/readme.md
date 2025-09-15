# Colliders

- HitboxCollector：
  - 递归收集 PolygonCollider2D（Mountain 或任意节点）
  - 统计信息与筛选（按名称、顶点数排序）
  - 导出顶点数据与世界坐标至 HitboxDump/

输入/输出：
- 输入：场景树的任意 GameObject
- 输出：控制台统计与文本导出

维护建议：
- 避免在 Update 中做全树扫描，尽量按需触发
- 导出前对路径与数量进行提示
