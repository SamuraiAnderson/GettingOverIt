# ColliderCollection - 碰撞体几何采集与可视化

## 文件列表

| 文件 | 说明 |
|------|------|
| **ColliderExporter.cs** | 静态工具类，将环境（Mountain）和 Player 轮廓的 PolygonCollider2D 顶点导出为 JSON |
| **ColliderVisualizer.cs** | GL 描边可视化，附加到主摄像机，实时绘制 Player 所有 Collider2D 轮廓 |

## 导出文件

运行时自动导出到 `GoiData/Colliders/`：

- `environment.json` — Mountain 下所有 PolygonCollider2D 的**世界坐标**顶点，保留多 path 结构
- `player_contour.json` — Player 三个关键碰撞部件（body / tip / pot / pot_sides）的**本地坐标**顶点

## Player 碰撞部件

| 部件 | GameObject 名称 | 功能 |
|------|-----------------|------|
| body | Player | 角色上半身碰撞轮廓 |
| tip | Tip | 锤头，核心交互部件 |
| pot | PotCollider | 锅底，承载角色重量 |
| pot_sides | Sides | 锅侧壁，独立碰撞响应 |
