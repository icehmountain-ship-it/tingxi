# 听席 · tingxi

浏览器端**说话人分离**与**自我声纹识别**。面向嘈杂场景：把喇叭、音乐、电视、环境噪声判为「不是你」；把其他人分到「说话人甲 / 乙 / …」席位；把已录入声纹识别为「你」。

声纹只保存在本机 `localStorage`，无账号、无数据库、无服务端模型下载。

## 功能

1. **音频门控** `classifyAudio`：在说话人识别前先分为 `silence | noise | music | speech`
2. **在线聚类** `OnlineSpeakerClusterer`：「你」阈值默认 `0.78`，其他人 `0.7`；另有 **认我保护带**（默认阈值下约 `0.66`）：轻声/近我帧不得新建「说话人甲」，高相似度时不因音高 alone 降成别人
3. **分席音高隔离**：他人席匹配要求对方明显赢过「你」，且音高相容，避免男女大跨度并席；席位标签附带男/女、大人/孩子等粗属性（未知允许）
4. **轻量声纹**：Mel + 基频特征，本地 FFT 提取，不下载大型 ML 模型
5. **界面**：录入 3 段声纹 → 实时听席 → 人数估计、拒绝原因、灵敏度滑杆、「不会当成你的」列表

## 技术栈

- Next.js App Router + TypeScript
- Tailwind CSS v4 + shadcn/ui
- 纯浏览器 Web Audio（麦克风）

## 本地运行

```bash
npm install
npm run dev
```

开发服务器默认端口：**3847** → [http://localhost:3847](http://localhost:3847)

```bash
npm run build   # 生产构建
npm run smoke   # 音频管线冒烟测试（无需麦克风）
```

## 目录要点

```
src/lib/audio/     # 采集、FFT、特征、VAD、聚类、引擎、localStorage
src/components/    # 听席主界面、波形、席位芯片、shadcn 控件
scripts/smoke-audio.ts
```

## 隐私

声纹向量仅写入浏览器 `localStorage`（键名 `tingxi.voiceprint.v1`）。清除浏览器数据或点击界面「清除」即可删除。

## 许可

私有项目；按需自行约定使用条款。
