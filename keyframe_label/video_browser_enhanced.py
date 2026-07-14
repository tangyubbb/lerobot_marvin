#!/usr/bin/env python3
"""
LeRobot Dataset Video Browser (增强版 - 支持 AV1 + 关键帧标注)

浏览数据集中的视频，支持：
- 拖动滑块播放视频
- 'p'/'n' 键切换上一个/下一个 episode
- 空格键暂停/播放
- 'k' 键标记当前帧为关键帧（每个 episode 只能标注一次）
- 's' 键保存当前 episode 的关键帧标注
- 'z' 键清除当前 episode 的所有标记（包括已保存的）
- 'q' 键退出
- 自动选择最佳视频解码器（PyAV 或 OpenCV）

标注规则：
- 每个 episode 只能完整标注一次
- 按 'k' 标记当前帧，再按 'k' 取消标记（仅当前 session）
- 已保存的帧无法通过 'k' 取消，需要用 'z' 清除所有标记后重新标注
- 按 's' 保存后，当前 episode 的标注即固定
- 按 'z' 可清除当前 episode 的所有标记（包括之前保存的），以便重新标注

用法：
    python keyframe_label/video_browser_enhanced.py --repo-id /home/marvin/hhw/peg_optical_module_0707 --camera right_close

示例：
    python video_browser_enhanced.py --repo-id hukewei/eval_pro_act5
    python video_browser_enhanced.py --repo-id hukewei/eval_pro_act5 --camera left_wrist --field-name is_keyframe
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 尝试导入 PyAV（可选依赖）
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    print("警告: PyAV 未安装，AV1 视频可能无法解码。建议安装: pip install av")


class VideoBrowser:
    def __init__(self, dataset_root: Path, camera_name: str | None = None, force_opencv: bool = False, field_name: str = "is_keyframe"):
        """
        初始化视频浏览器

        Args:
            dataset_root: 数据集根目录
            camera_name: 相机名称，如果为None则使用第一个相机
            force_opencv: 强制使用 OpenCV（即使 PyAV 可用）
            field_name: 标签字段名称
        """
        self.dataset_root = Path(dataset_root)
        self.info_path = self.dataset_root / "meta" / "info.json"
        self.force_opencv = force_opencv
        self.field_name = field_name
        self.labels_dir = self.dataset_root / "labels"

        # 标注状态
        self.keyframe_marks = {}  # {episode_index: set(frame_indices)}
        self.current_episode_marks = set()  # 当前 episode 的标记（仅当前session）
        self.current_episode_saved_marks = set()  # 当前 episode 已保存的标记
        self.labels_df = None
        self.labels_modified = False

        # 加载数据集信息
        with open(self.info_path, 'r') as f:
            self.info = json.load(f)

        # 获取所有视频特征
        self.video_keys = [
            key.replace("observation.images.", "")
            for key in self.info["features"].keys()
            if self.info["features"][key].get("dtype") == "video"
        ]

        if not self.video_keys:
            raise ValueError("数据集中没有找到视频数据")

        # 选择相机
        if camera_name is None:
            self.camera_name = self.video_keys[0]
            print(f"未指定相机，使用默认相机: {self.camera_name}")
        elif camera_name in self.video_keys:
            self.camera_name = camera_name
        else:
            raise ValueError(
                f"相机 '{camera_name}' 不存在。可用的相机: {', '.join(self.video_keys)}"
            )

        print(f"可用相机: {', '.join(self.video_keys)}")
        print(f"当前相机: {self.camera_name}")

        # 获取视频路径模板
        self.video_path_template = self.info["video_path"]
        self.video_key_path = f"observation.images.{self.camera_name}"

        # 初始化状态
        self.total_episodes = self.info["total_episodes"]
        self.current_episode = 0
        self.current_frame = 0
        self.is_playing = False
        self.fps = self.info["fps"]

        # 视频解码器状态
        self.container = None
        self.video_stream = None
        self.cap = None
        self.use_pyav = False

        # 加载episode元数据
        self._load_episode_metadata()

        # 加载或创建标签文件
        self._load_or_create_labels()

        # 加载第一个视频
        self.load_episode(0)

    def _load_or_create_labels(self):
        """加载或创建标签文件"""
        self.labels_dir.mkdir(exist_ok=True)
        label_path = self.labels_dir / f"{self.field_name}.parquet"

        if label_path.exists():
            self.labels_df = pd.read_parquet(label_path)
            print(f"已加载标签文件: {self.field_name}")

            # 加载所有 episode 的已保存标记（用于统计）
            for ep_idx in self.labels_df['episode_index'].unique():
                ep_data = self.labels_df[self.labels_df['episode_index'] == ep_idx]
                marked_frames = set(ep_data[ep_data[self.field_name] == 1]['frame_index'].values)
                if marked_frames:
                    self.keyframe_marks[ep_idx] = marked_frames
        else:
            print(f"标签文件不存在: {label_path}")
            print(f"将在首次保存时创建")
            self.labels_df = None

    def _load_episode_metadata(self):
        """加载所有episode的元数据"""
        episodes_dir = self.dataset_root / "meta" / "episodes"
        self.episode_metadata = []

        # 读取所有episode元数据文件
        for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
            for meta_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(meta_file)
                self.episode_metadata.append(df)

        if self.episode_metadata:
            self.episode_metadata = pd.concat(self.episode_metadata, ignore_index=True)
            print(f"加载了 {len(self.episode_metadata)} 个episode的元数据")
        else:
            print("警告: 未找到episode元数据，将使用简化的视频查找方式")

    def get_video_path(self, episode_index: int) -> tuple[Path, int, int]:
        """
        获取指定episode的视频文件路径和时间范围

        Args:
            episode_index: Episode索引

        Returns:
            (video_path, from_frame, to_frame): 视频路径和该episode在视频中的帧范围
        """
        if hasattr(self, 'episode_metadata') and len(self.episode_metadata) > 0:
            # 使用episode元数据查找
            ep_meta = self.episode_metadata[
                self.episode_metadata['episode_index'] == episode_index
            ]

            if len(ep_meta) == 0:
                raise ValueError(f"未找到Episode {episode_index}的元数据")

            ep_meta = ep_meta.iloc[0]

            # 获取视频chunk和file索引
            video_col_prefix = f"videos/{self.video_key_path}"
            chunk_index = ep_meta[f"{video_col_prefix}/chunk_index"]
            file_index = ep_meta[f"{video_col_prefix}/file_index"]

            # 构造视频路径
            video_path = self.dataset_root / "videos" / self.video_key_path / \
                        f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"

            # 获取该episode在视频中的帧范围（通过timestamp转换）
            from_timestamp = ep_meta[f"{video_col_prefix}/from_timestamp"]
            to_timestamp = ep_meta[f"{video_col_prefix}/to_timestamp"]

            from_frame = int(from_timestamp * self.fps)
            to_frame = int(to_timestamp * self.fps)

            return video_path, from_frame, to_frame
        else:
            # 简化策略：查找存在的视频文件
            video_dir = self.dataset_root / "videos" / self.video_key_path

            for chunk_dir in sorted(video_dir.glob("chunk-*")):
                for idx, video_file in enumerate(sorted(chunk_dir.glob("file-*.mp4"))):
                    if idx == episode_index:
                        # 假设整个视频文件对应一个episode
                        cap = cv2.VideoCapture(str(video_file))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        return video_file, 0, total_frames

            raise FileNotFoundError(f"未找到Episode {episode_index}的视频文件")

    def _open_with_pyav(self, video_path: Path) -> bool:
        """尝试使用 PyAV 打开视频"""
        if not PYAV_AVAILABLE or self.force_opencv:
            return False

        try:
            self.container = av.open(str(video_path))
            self.video_stream = self.container.streams.video[0]
            # 强制使用软件解码（避免硬件加速问题）
            self.video_stream.codec_context.thread_type = 'AUTO'
            return True
        except Exception as e:
            print(f"  PyAV 打开失败: {e}")
            if self.container:
                self.container.close()
                self.container = None
            return False

    def _open_with_opencv(self, video_path: Path) -> bool:
        """使用 OpenCV 打开视频"""
        self.cap = cv2.VideoCapture(str(video_path))
        return self.cap.isOpened()

    def load_episode(self, episode_index: int) -> bool:
        """
        加载指定的episode视频

        Args:
            episode_index: Episode索引

        Returns:
            是否成功加载
        """
        if episode_index < 0 or episode_index >= self.total_episodes:
            print(f"Episode {episode_index} 超出范围 [0, {self.total_episodes-1}]")
            return False

        try:
            video_path, from_frame, to_frame = self.get_video_path(episode_index)
        except (ValueError, FileNotFoundError) as e:
            print(f"错误: {e}")
            return False

        if not video_path.exists():
            print(f"视频文件不存在: {video_path}")
            return False

        # 释放之前的资源
        if self.container is not None:
            self.container.close()
            self.container = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # 尝试打开视频（优先 PyAV，失败则 OpenCV）
        self.use_pyav = self._open_with_pyav(video_path)

        if not self.use_pyav:
            if not self._open_with_opencv(video_path):
                print(f"无法打开视频: {video_path}")
                return False

        # 设置episode参数
        self.current_episode = episode_index
        self.episode_from_frame = from_frame
        self.episode_to_frame = to_frame
        self.total_frames = to_frame - from_frame
        self.current_frame = 0

        # 加载当前 episode 的已保存标记
        self.current_episode_marks = set()
        self.current_episode_saved_marks = set()
        if episode_index in self.keyframe_marks:
            self.current_episode_saved_marks = self.keyframe_marks[episode_index].copy()
            print(f"  发现已保存的标记: {len(self.current_episode_saved_marks)} 个关键帧")

        # 初始化帧缓存（PyAV使用）
        if self.use_pyav:
            self._frame_cache = []
            self._cache_start_frame = from_frame
            self._preload_frames()

        episode_duration = self.total_frames / self.fps

        print(f"\n加载 Episode {episode_index}: {video_path.name}")
        print(f"  视频文件帧范围: [{from_frame}, {to_frame})")
        print(f"  Episode帧数: {self.total_frames}")
        print(f"  解码器: {'PyAV (软件解码AV1)' if self.use_pyav else 'OpenCV'}")
        print(f"  时长: {episode_duration:.2f} 秒")

        return True

    def _preload_frames(self):
        """预加载帧到缓存（仅PyAV）"""
        if not self.use_pyav:
            return

        print(f"  正在解码 episode 帧...", end='', flush=True)

        self._frame_cache = []
        frame_count = 0

        try:
            # 跳到起始位置
            target_pts = int(self.episode_from_frame / self.fps / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream)

            # 解码所有帧
            for packet in self.container.demux(self.video_stream):
                for frame in packet.decode():
                    # 转换为 numpy array (BGR 格式用于 OpenCV 显示)
                    img = frame.to_ndarray(format='bgr24')
                    self._frame_cache.append(img)
                    frame_count += 1

                    if frame_count >= self.total_frames:
                        break

                if frame_count >= self.total_frames:
                    break

            print(f" 完成 ({frame_count} 帧)")

        except Exception as e:
            print(f"\n  警告: 解码过程中出现错误: {e}")
            print(f"  已缓存 {len(self._frame_cache)} 帧")

    def seek_frame(self, frame_index: int):
        """跳转到指定帧（相对于episode开始）"""
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame = frame_index

        if not self.use_pyav:
            # OpenCV: 设置绝对帧位置
            absolute_frame = self.episode_from_frame + frame_index
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)

    def read_frame(self) -> np.ndarray | None:
        """读取当前帧"""
        if self.current_frame >= self.total_frames:
            return None

        if self.use_pyav:
            # PyAV: 从缓存读取
            if self.current_frame < len(self._frame_cache):
                return self._frame_cache[self.current_frame]
            else:
                return None
        else:
            # OpenCV: 实时读取
            absolute_frame = self.episode_from_frame + self.current_frame
            if absolute_frame >= self.episode_to_frame:
                return None

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None

    def on_trackbar_change(self, value: int):
        """滑块回调函数"""
        self.seek_frame(value)

    def mark_keyframe(self):
        """标记当前帧为关键帧（当前 episode 仅能标记一次）"""
        frame = self.current_frame

        # 检查是否已经有标记（包括已保存的）
        if frame in self.current_episode_saved_marks:
            print(f"⚠ Frame {frame} 已在之前保存过，无法重复标记")
            print(f"   如需重新标注此 episode，请先按 'z' 清除所有标记")
            return

        if frame in self.current_episode_marks:
            # 当前 session 已标记，取消标记
            self.current_episode_marks.remove(frame)
            print(f"✗ 取消标记 Frame {frame}")
        else:
            # 添加标记
            self.current_episode_marks.add(frame)
            print(f"✓ 标记关键帧 Frame {frame}")

        self.labels_modified = True

    def undo_mark(self):
        """清除当前 episode 的所有标记（包括已保存的）"""
        ep = self.current_episode

        # 统计要清除的标记数量
        current_marks_count = len(self.current_episode_marks)
        saved_marks_count = len(self.current_episode_saved_marks)
        total_count = current_marks_count + saved_marks_count

        if total_count == 0:
            print("当前 Episode 没有标记")
            return

        # 清除当前 session 的标记
        self.current_episode_marks.clear()

        # 如果有已保存的标记，需要从数据库中删除
        if saved_marks_count > 0:
            if self.labels_df is not None:
                # 将当前 episode 的所有标记设为 0
                mask = self.labels_df['episode_index'] == ep
                self.labels_df.loc[mask, self.field_name] = 0

                # 更新全局标记字典
                if ep in self.keyframe_marks:
                    del self.keyframe_marks[ep]

            self.current_episode_saved_marks.clear()

        print(f"↶ 已清除 Episode {ep} 的所有标记:")
        print(f"  - 当前 session 标记: {current_marks_count} 个")
        print(f"  - 已保存的标记: {saved_marks_count} 个")
        print(f"  - 总计清除: {total_count} 个")

        self.labels_modified = True

    def save_labels(self):
        """保存当前 episode 的标签"""
        if not self.labels_modified:
            print("没有修改，无需保存")
            return

        # 如果标签文件不存在，需要先创建
        if self.labels_df is None:
            print("标签文件不存在，请先运行 add_frame_labels.py 创建标签文件")
            return

        ep = self.current_episode

        # 先清除当前 episode 的所有标记（设为 0）
        mask = self.labels_df['episode_index'] == ep
        self.labels_df.loc[mask, self.field_name] = 0

        # 更新当前 episode 的标记
        for frame in self.current_episode_marks:
            mask = (self.labels_df['episode_index'] == ep) & \
                   (self.labels_df['frame_index'] == frame)
            self.labels_df.loc[mask, self.field_name] = 1

        # 更新全局标记字典
        if len(self.current_episode_marks) > 0:
            self.keyframe_marks[ep] = self.current_episode_marks.copy()
        elif ep in self.keyframe_marks:
            del self.keyframe_marks[ep]

        # 保存到文件
        label_path = self.labels_dir / f"{self.field_name}.parquet"
        self.labels_df.to_parquet(label_path, index=False)

        # 统计
        ep_marks = len(self.current_episode_marks)
        total_marks = sum(len(frames) for frames in self.keyframe_marks.values())

        print(f"\n✓ 标签已保存: {label_path}")
        print(f"  当前 Episode {self.current_episode}: {ep_marks} 个关键帧")
        print(f"  全部关键帧: {total_marks} 个")

        # 保存后，将当前标记转移到已保存标记中，并清空当前标记
        self.current_episode_saved_marks = self.current_episode_marks.copy()
        self.current_episode_marks.clear()

        self.labels_modified = False

    def is_marked_keyframe(self, episode: int, frame: int) -> bool:
        """检查指定帧是否为关键帧（包括当前 session 和已保存的）"""
        if episode != self.current_episode:
            # 其他 episode，只检查全局字典
            return episode in self.keyframe_marks and frame in self.keyframe_marks[episode]
        else:
            # 当前 episode，检查当前 session 和已保存的标记
            return (frame in self.current_episode_marks or
                    frame in self.current_episode_saved_marks)

    def run(self):
        """运行视频浏览器"""
        window_name = f"LeRobot Video Browser - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # 创建滑块
        trackbar_name = "Frame"
        cv2.createTrackbar(trackbar_name, window_name, 0,
                          max(self.total_frames - 1, 1),
                          self.on_trackbar_change)

        print("\n" + "="*60)
        print("控制说明:")
        print("  滑块: 拖动播放")
        print("  空格: 暂停/播放")
        print("  p: 上一个episode")
        print("  n: 下一个episode")
        print("  ←/→: 前一帧/后一帧")
        print("  k: 标记/取消标记当前帧为关键帧")
        print("  s: 保存当前 episode 的标签")
        print("  z: 清除当前 episode 的所有标记（包括已保存的）")
        print("  q/ESC: 退出")
        print("="*60)
        print("标注规则:")
        print("  - 每个 episode 只能完整标注一次")
        print("  - 已保存的帧无法单独取消，需用 'z' 清除后重新标注")
        print("  - 按 'z' 可清除当前 episode 的所有标记以便重新标注")
        print("="*60 + "\n")

        while True:
            # 读取并显示当前帧
            frame = self.read_frame()

            if frame is None:
                # 视频结束，重新开始
                self.seek_frame(0)
                frame = self.read_frame()
                if frame is None:
                    print("无法读取视频帧")
                    break

            # 在帧上添加信息
            info_frame = frame.copy()

            # 添加文本信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            timestamp = self.current_frame / self.fps

            # 检查当前帧是否为关键帧
            is_keyframe = self.is_marked_keyframe(self.current_episode, self.current_frame)

            # 统计关键帧数量（当前 session + 已保存）
            current_marks = len(self.current_episode_marks)
            saved_marks = len(self.current_episode_saved_marks)
            ep_keyframes = current_marks + saved_marks

            # 统计已标注的episode数量
            annotated_episodes = len(self.keyframe_marks)

            # 检测漏标注的episode（当前episode之前未标注的）
            missing_episodes = []
            for ep_idx in range(self.current_episode):
                if ep_idx not in self.keyframe_marks:
                    missing_episodes.append(ep_idx)

            # 格式化漏标注信息
            if len(missing_episodes) == 0:
                missing_info = "无"
            elif len(missing_episodes) <= 3:
                missing_info = ", ".join([f"Ep{ep}" for ep in missing_episodes])
            else:
                # 超过3个，显示前3个加...
                first_three = ", ".join([f"Ep{ep}" for ep in missing_episodes[:3]])
                missing_info = f"{first_three} ... (共{len(missing_episodes)}个)"

            texts = [
                f"Episode: {self.current_episode}/{self.total_episodes-1}",
                f"Frame: {self.current_frame}/{self.total_frames-1}",
                f"Time: {timestamp:.2f}s",
                f"Camera: {self.camera_name}",
                f"Decoder: {'PyAV' if self.use_pyav else 'OpenCV'}",
                f"Status: {'Playing' if self.is_playing else 'Paused'}",
                f"Annotated: {annotated_episodes}/{self.total_episodes}",
                f"Missing: {missing_info}",
                f"Keyframes: {ep_keyframes} (New:{current_marks} Saved:{saved_marks})",
                f"Current: {'KEYFRAME' if is_keyframe else 'Normal'}"
            ]

            y_offset = 30
            for i, text in enumerate(texts):
                # 关键帧状态使用不同颜色
                if i == len(texts) - 1 and is_keyframe:
                    color = (0, 0, 255)  # 红色表示关键帧
                else:
                    color = (0, 255, 0)  # 绿色

                cv2.putText(info_frame, text, (10, y_offset),
                           font, 0.6, color, 2, cv2.LINE_AA)
                y_offset += 25

            # 如果是关键帧，在画面上添加明显标记
            if is_keyframe:
                h, w = info_frame.shape[:2]
                # 绘制边框
                cv2.rectangle(info_frame, (5, 5), (w-5, h-5), (0, 0, 255), 5)
                # 添加"KEY"标记
                cv2.putText(info_frame, "KEY", (w-150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(window_name, info_frame)

            # 更新滑块位置
            cv2.setTrackbarPos(trackbar_name, window_name, self.current_frame)

            # 根据播放状态决定等待时间
            if self.is_playing:
                wait_time = int(1000 / self.fps)
            else:
                wait_time = 10  # 暂停时快速响应

            # 处理按键
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:  # q 或 ESC
                # 退出前提示是否保存
                if self.labels_modified:
                    print("\n警告: 有未保存的修改！")
                    print("按 's' 保存后再退出，或按 'q' 强制退出")
                    continue
                break
            elif key == ord(' '):  # 空格：暂停/播放
                self.is_playing = not self.is_playing
                status = "Playing" if self.is_playing else "Paused"
                print(f"Status: {status}")
            elif key == ord('k'):  # k：标记/取消标记关键帧
                self.is_playing = False  # 暂停播放
                self.mark_keyframe()
            elif key == ord('s'):  # s：保存标签
                self.save_labels()
            elif key == ord('z'):  # z：撤销
                self.undo_mark()
            elif key == ord('p'):  # p：上一个episode
                self.is_playing = False
                if self.load_episode(self.current_episode - 1):
                    cv2.setTrackbarMax(trackbar_name, window_name,
                                      max(self.total_frames - 1, 1))
            elif key == ord('n'):  # n：下一个episode
                self.is_playing = False
                if self.load_episode(self.current_episode + 1):
                    cv2.setTrackbarMax(trackbar_name, window_name,
                                      max(self.total_frames - 1, 1))
            elif key == 81 or key == 2:  # 左箭头
                self.is_playing = False
                self.seek_frame(self.current_frame - 1)
            elif key == 83 or key == 3:  # 右箭头
                self.is_playing = False
                self.seek_frame(self.current_frame + 1)

            # 如果在播放状态，自动前进
            if self.is_playing:
                if self.current_frame >= self.total_frames - 1:
                    # 视频结束，停止播放
                    self.is_playing = False
                    self.seek_frame(0)
                else:
                    self.current_frame += 1

        # 清理
        if self.container:
            self.container.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot Dataset Video Browser (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python video_browser_enhanced.py --repo-id hukewei/eval_pro_act5
  python video_browser_enhanced.py --repo-id hukewei/eval_pro_act5 --camera left_wrist
  python video_browser_enhanced.py --root /path/to/dataset --camera right_eye
  python video_browser_enhanced.py --root /path/to/dataset --force-opencv
        """
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="数据集ID (格式: username/dataset_name)"
    )

    parser.add_argument(
        "--root",
        type=str,
        help="数据集根目录路径（如果不使用repo-id）"
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="相机名称（如 'left_wrist', 'right_eye'）。如果不指定，使用第一个相机"
    )

    parser.add_argument(
        "--field-name",
        type=str,
        default="is_keyframe",
        help="标签字段名称（默认为 'is_keyframe'）"
    )

    parser.add_argument(
        "--force-opencv",
        action="store_true",
        help="强制使用 OpenCV（即使 PyAV 可用）"
    )

    args = parser.parse_args()

    # 确定数据集路径
    if args.root:
        dataset_root = Path(args.root)
    elif args.repo_id:
        # 默认路径：~/.cache/huggingface/lerobot/{repo_id}
        home = Path.home()
        dataset_root = home / ".cache" / "huggingface" / "lerobot" / args.repo_id
    else:
        parser.error("必须指定 --repo-id 或 --root")

    if not dataset_root.exists():
        print(f"错误: 数据集路径不存在: {dataset_root}")
        return

    # 创建并运行浏览器
    try:
        browser = VideoBrowser(
            dataset_root,
            args.camera,
            force_opencv=args.force_opencv,
            field_name=args.field_name
        )
        browser.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
