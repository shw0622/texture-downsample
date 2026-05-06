# textures/

本目录下的真实 PBR 纹理素材**不入 git 仓库**（体积太大），需自行下载。

## v3.1 验证使用的 5 张

均来自 [ambientCG.com](https://ambientcg.com/)（CC0 许可），下载 2K-PNG 版本即可：

| 名称 | 类别（factor=4 路由判定） | 下载链接 |
|---|---|---|
| Bricks104 | GEOMETRIC | https://ambientcg.com/view?id=Bricks104 |
| ChristmasTreeOrnament019 | METAL_FLAT | https://ambientcg.com/view?id=ChristmasTreeOrnament019 |
| Metal007 | METAL_FLAT | https://ambientcg.com/view?id=Metal007 |
| Metal049A | CONSTANT | https://ambientcg.com/view?id=Metal049A |
| Metal053C | METAL_FLAT | https://ambientcg.com/view?id=Metal053C |

## 目录结构约定

每张材质放在 `textures/<NAME>_2K-PNG/` 下，文件名按 ambientCG 默认命名：

```
textures/
└── Bricks104_2K-PNG/
    ├── Bricks104_2K-PNG_Color.png            # albedo (sRGB)
    ├── Bricks104_2K-PNG_NormalGL.png         # normal (OpenGL 约定)
    ├── Bricks104_2K-PNG_Roughness.png        # roughness (linear)
    ├── Bricks104_2K-PNG_Metalness.png        # metallic (二值)，可选
    └── Bricks104_2K-PNG_AmbientOcclusion.png # AO，可选
```

加载逻辑见 `examples/verify_v3_native4x_routed.py::load_set`：
按文件名后缀匹配 `_Color` / `_NormalGL` / `_Roughness` / `_Metalness|_Metallic` / `_AmbientOcclusion|_AO`。
