import streamlit as st
import tempfile
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import subprocess
import os
import json
import zlib
import numpy as np
import gc
import concurrent.futures
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SoulGenesis - Ultimate Media Compression",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Utility Functions
# -----------------------------
@st.cache_resource
def load_platform_config():
    """Load platform configuration from JSON file"""
    try:
        with open('platform_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("platform_config.json not found. Using default configuration.")
        return {
            "custom": {
                "video": {"crf": 28, "preset": "medium", "max_height": 1080, 
                         "audio_bitrate": "128k", "target_size_mb": None, "codec": "libx264"},
                "image": {"quality": 85, "use_palette": True, "target_size_kb": None}
            }
        }

def cleanup_temp_files(*file_paths):
    """Clean up temporary files and clear memory"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
    gc.collect()

def check_ffmpeg():
    """Check if FFmpeg is available on the system and what codecs are supported"""
    try:
        # Check if ffmpeg exists
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "FFmpeg not found"
        
        # Check available codecs
        codecs_result = subprocess.run(["ffmpeg", "-codecs"], 
                                     capture_output=True, text=True, timeout=10)
        codecs_output = codecs_result.stdout.lower()
        
        supported_codecs = []
        if "libx264" in codecs_output:
            supported_codecs.append("libx264")
        if "libx265" in codecs_output:
            supported_codecs.append("libx265")
        
        return True, supported_codecs
        
    except FileNotFoundError:
        return False, "FFmpeg command not found"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out"
    except Exception as e:
        return False, f"FFmpeg check error: {str(e)}"

def get_file_type(file_name):
    """Determine if file is image or video based on extension"""
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']
    video_exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
    
    ext = Path(file_name).suffix.lower()
    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    else:
        return "unknown"

def get_video_duration(input_path):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 60.0  # Default assumption if cannot get duration

# -----------------------------
# Watermark Functions
# -----------------------------
def add_watermark_to_image(input_image, watermark_image, position='bottom-right', opacity=0.7, scale=0.2):
    """Add watermark to a PIL Image"""
    # Ensure watermark has alpha channel
    if watermark_image.mode != 'RGBA':
        watermark_image = watermark_image.convert('RGBA')
    
    # Resize watermark
    base_width = input_image.width
    wm_size = int(base_width * scale)
    watermark_image = watermark_image.resize((wm_size, wm_size), Image.Resampling.LANCZOS)
    
    # Adjust opacity
    alpha = watermark_image.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    watermark_image.putalpha(alpha)
    
    # Create transparent layer for watermark
    transparent = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
    
    # Calculate position
    if position == 'bottom-right':
        wm_position = (input_image.width - watermark_image.width - 10, 
                       input_image.height - watermark_image.height - 10)
    elif position == 'top-right':
        wm_position = (input_image.width - watermark_image.width - 10, 10)
    elif position == 'bottom-left':
        wm_position = (10, input_image.height - watermark_image.height - 10)
    elif position == 'top-left':
        wm_position = (10, 10)
    elif position == 'center':
        wm_position = ((input_image.width - watermark_image.width) // 2,
                       (input_image.height - watermark_image.height) // 2)
    else:
        wm_position = (10, 10)
    
    # Paste watermark
    transparent.paste(watermark_image, wm_position, watermark_image)
    
    # Convert base image to RGBA and composite with watermark
    if input_image.mode != 'RGBA':
        input_image = input_image.convert('RGBA')
    
    return Image.alpha_composite(input_image, transparent)

# -----------------------------
# Compression Functions
# -----------------------------
def compress_image(input_path, output_path, config):
    """Compress image using adaptive method based on config"""
    img = Image.open(input_path).convert("RGB")
    original_size = os.path.getsize(input_path) / 1024  # KB
    
    # Apply compression based on config
    quality = config.get('quality', 85)
    use_palette = config.get('use_palette', True)
    target_size_kb = config.get('target_size_kb')
    
    # If target size is specified, adjust quality to meet it
    if target_size_kb:
        # Simple binary search for appropriate quality
        low, high = 10, 95
        for _ in range(5):  # 5 iterations should be enough
            mid = (low + high) // 2
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name, quality=mid, optimize=True)
                tmp_size = os.path.getsize(tmp.name) / 1024
                os.unlink(tmp.name)
            
            if tmp_size > target_size_kb:
                high = mid - 1  # Need lower quality
            else:
                low = mid + 1  # Can try higher quality
        
        quality = max(10, min(95, (low + high) // 2))
    
    # Choose compression method
    if use_palette and (img.size[0] * img.size[1] <= 500000):
        # Use palette method for smaller images
        img_p = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        palette = img_p.getpalette()[:768]
        arr = np.array(img_p)
        data = {
            "size": img.size,
            "palette": [palette[i:i+3] for i in range(0, len(palette), 3)],
            "pixels": arr.flatten().tolist()
        }
        compressed_data = zlib.compress(json.dumps(data).encode("utf-8"), level=9)
        method = "Palette+Zlib"
        
        with open(output_path, "wb") as f:
            f.write(compressed_data)
    else:
        # Use standard JPEG compression
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, quality=quality, optimize=True)
            tmp.seek(0)
            compressed_data = zlib.compress(tmp.read(), level=9)
            method = "JPEG+Zlib"
            os.unlink(tmp.name)
        
        with open(output_path, "wb") as f:
            f.write(compressed_data)
    
    return method, quality

def decompress_image(input_path, output_path):
    """Decompress .genesis file back to image"""
    with open(input_path, "rb") as f:
        data = zlib.decompress(f.read())
    
    try:
        # Try to read as standard compressed image
        with open(output_path, "wb") as f:
            f.write(data)
        Image.open(output_path).verify()
    except:
        # Try palette method
        try:
            data = json.loads(data.decode("utf-8"))
            size = tuple(data["size"])
            palette = data["palette"]
            pixels = np.array(data["pixels"], dtype=np.uint8).reshape(size[1], size[0])
            img_p = Image.fromarray(pixels, mode="P")
            img_p.putpalette([x for rgb in palette for x in rgb])
            img_rgb = img_p.convert("RGB")
            img_rgb.save(output_path)
        except:
            raise ValueError("Invalid .genesis file format")

def compress_video_safe(in_path, out_path, config):
    """Safe video compression with fallback options"""
    in_path = str(Path(in_path))
    out_path = str(Path(out_path))

    # Get available codecs
    ffmpeg_available, codec_info = check_ffmpeg()
    if not ffmpeg_available:
        raise Exception("FFmpeg not available: " + str(codec_info))
    
    supported_codecs = codec_info if isinstance(codec_info, list) else []
    
    # Use configuration with fallbacks
    crf = config.get('crf', 28)
    preset = config.get('preset', 'medium')
    max_height = config.get('max_height', 1080)
    audio_bitrate = config.get('audio_bitrate', '128k')
    target_size_mb = config.get('target_size_mb')
    preferred_codec = config.get('codec', 'libx264')
    
    # Choose available codec
    if preferred_codec in supported_codecs:
        video_codec = preferred_codec
    elif 'libx264' in supported_codecs:
        video_codec = 'libx264'
        st.warning("H.265 not available, using H.264 instead")
    else:
        raise Exception("No supported video codecs found. Available: " + str(supported_codecs))
    
    # Build basic ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", video_codec,
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        "-loglevel", "error",
    ]

    # Add scale filter if max_height is specified
    if max_height:
        cmd.extend(["-vf", f"scale=-2:{max_height}"])

    cmd.append(out_path)

    # For target size, use single-pass with bitrate estimation (more reliable)
    if target_size_mb:
        duration = get_video_duration(in_path)
        target_bitrate = int((target_size_mb * 8192) / max(1, duration))  # MB * 8192 = kbit
        
        # Replace CRF with bitrate control
        cmd[cmd.index("-crf") + 1] = "-b:v"
        cmd.insert(cmd.index("-b:v") + 1, f"{target_bitrate}k")
        cmd.insert(cmd.index("-b:v") + 2, "-maxrate")
        cmd.insert(cmd.index("-maxrate") + 1, f"{target_bitrate}k")
        cmd.insert(cmd.index("-maxrate") + 2, "-bufsize")
        cmd.insert(cmd.index("-bufsize") + 1, f"{target_bitrate * 2}k")
    
    # Run the command with timeout
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise Exception(f"FFmpeg error: {error_msg}")
    except subprocess.TimeoutExpired:
        raise Exception("Compression timed out after 5 minutes")
    except Exception as e:
        raise Exception(f"Compression failed: {str(e)}")

def decompress_video(in_path, out_path):
    """Decompress video by re-encoding to H.264"""
    in_path = str(Path(in_path))
    out_path = str(Path(out_path))

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-crf", "23",  # Slightly higher CRF for decompression
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        out_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise Exception(f"FFmpeg error: {error_msg}")
    except subprocess.TimeoutExpired:
        raise Exception("Decompression timed out after 5 minutes")

# -----------------------------
# UI Components
# -----------------------------
def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.title("⚙️ SoulGenesis Settings")
    
    # Check FFmpeg status first
    ffmpeg_available, codec_info = check_ffmpeg()
    
    if not ffmpeg_available:
        st.sidebar.error(f"⚠️ FFmpeg Issue: {codec_info}")
        st.sidebar.info("Video compression will not work. Image compression is available.")
    else:
        st.sidebar.success(f"✅ FFmpeg Available")
        if isinstance(codec_info, list):
            st.sidebar.info(f"Supported codecs: {', '.join(codec_info)}")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Single File", "Batch Processing"],
        help="Choose to process a single file or multiple files"
    )
    
    # Platform selection
    platform_config = load_platform_config()
    platforms = list(platform_config.keys())
    platform = st.sidebar.selectbox(
        "Optimize for Platform",
        platforms,
        help="Select the target platform for optimization"
    )
    
    # Watermark settings
    st.sidebar.markdown("---")
    add_watermark = st.sidebar.checkbox("Add Watermark", False)
    watermark_options = None
    
    if add_watermark:
        watermark_file = st.sidebar.file_uploader("Upload Watermark Image", type=['png', 'jpg', 'jpeg'])
        if watermark_file:
            watermark_position = st.sidebar.selectbox(
                "Watermark Position",
                ['bottom-right', 'top-right', 'bottom-left', 'top-left', 'center']
            )
            opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7)
            scale = st.sidebar.slider("Scale", 0.05, 0.5, 0.2)
            
            watermark_options = {
                'image': Image.open(watermark_file),
                'position': watermark_position,
                'opacity': opacity,
                'scale': scale
            }
    
    return mode, platform, watermark_options, ffmpeg_available

# -----------------------------
# Main Application
# -----------------------------
def main():
    # Initialize session state
    if "compression_count" not in st.session_state:
        st.session_state.compression_count = 0
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    
    # Header
    st.title("✨ SoulGenesis - Ultimate Media Compression")
    st.markdown("**Smart compression optimized for social media platforms • 100% offline processing**")
    
    # Sidebar
    mode, platform, watermark_options, ffmpeg_available = render_sidebar()
    
    # Main content
    if mode == "Single File":
        render_single_file_mode(platform, watermark_options, ffmpeg_available)
    else:
        render_batch_mode(platform, watermark_options, ffmpeg_available)
    
    # Footer
    st.markdown("---")
    st.markdown("**SoulGenesis Media Compressor** • Professional compression • Made for South African Hustlers")

def render_single_file_mode(platform, watermark_options, ffmpeg_available):
    """Render single file compression interface"""
    tab1, tab2 = st.tabs(["📤 Compress", "📥 Decompress"])
    
    with tab1:
        st.header("Compress Media File")
        
        uploaded_file = st.file_uploader(
            "Choose a file to compress",
            type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "mkv", "webm"],
            key="file_upload"
        )
        
        if uploaded_file:
            file_type = get_file_type(uploaded_file.name)
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if file_type == "image":
                    st.image(uploaded_file, caption="Original Image", use_container_width=True)
                elif file_type == "video":
                    st.video(uploaded_file)
                
                st.metric("Original Size", f"{file_size_mb:.2f} MB")
                st.metric("File Type", file_type.upper())
            
            if st.button("🚀 Compress File", type="primary", use_container_width=True):
                if file_type == "video" and not ffmpeg_available:
                    st.error("FFmpeg not available. Video compression requires FFmpeg.")
                    return
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_in:
                    tmp_in.write(uploaded_file.read())
                    tmp_in_path = tmp_in.name
                
                config = load_platform_config().get(platform, {}).get(file_type, {})
                
                with st.spinner("Compressing..."):
                    try:
                        if file_type == "image":
                            out_path = Path(tempfile.gettempdir()) / f"compressed_{Path(uploaded_file.name).stem}.genesis"
                            method, quality = compress_image(tmp_in_path, out_path, config)
                            
                            # Apply watermark if needed
                            if watermark_options:
                                img = Image.open(out_path)
                                watermarked_img = add_watermark_to_image(
                                    img, 
                                    watermark_options['image'],
                                    watermark_options['position'],
                                    watermark_options['opacity'],
                                    watermark_options['scale']
                                )
                                watermarked_img.save(out_path)
                            
                            compressed_size = os.path.getsize(out_path) / 1024
                            compression_ratio = (1 - compressed_size/(file_size/1024)) * 100
                            
                            with col2:
                                st.success("Compression Complete!")
                                st.metric("Compressed Size", f"{compressed_size:.1f} KB")
                                st.metric("Size Reduction", f"{compression_ratio:.1f}%")
                                st.info(f"Method: {method}, Quality: {quality}")
                            
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Compressed File",
                                    f.read(),
                                    file_name=Path(out_path).name,
                                    mime="application/octet-stream"
                                )
                        
                        elif file_type == "video":
                            out_path = Path(tempfile.gettempdir()) / f"compressed_{Path(uploaded_file.name).stem}.mp4"
                            compress_video_safe(tmp_in_path, out_path, config)
                            
                            compressed_size = os.path.getsize(out_path) / (1024 * 1024)
                            compression_ratio = (1 - compressed_size/file_size_mb) * 100
                            
                            with col2:
                                st.success("Compression Complete!")
                                st.metric("Compressed Size", f"{compressed_size:.2f} MB")
                                st.metric("Size Reduction", f"{compression_ratio:.1f}%")
                            
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Compressed Video",
                                    f.read(),
                                    file_name=Path(out_path).name,
                                    mime="video/mp4"
                                )
                        
                        st.session_state.compression_count += 1
                        cleanup_temp_files(tmp_in_path, out_path)
                        
                    except Exception as e:
                        st.error(f"Compression failed: {str(e)}")
                        st.info("This often happens when FFmpeg is not properly installed or doesn't support the required codecs.")
                        cleanup_temp_files(tmp_in_path)
    
    with tab2:
        st.header("Decompress File")
        genesis_file = st.file_uploader("Choose a .genesis file", type=["genesis"], key="decompress")
        
        if genesis_file:
            if st.button("🔍 Reconstruct File", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".genesis") as tmp_in:
                    tmp_in.write(genesis_file.read())
                    tmp_in_path = tmp_in.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_out:
                    tmp_out_path = tmp_out.name
                
                try:
                    with st.spinner("Reconstructing..."):
                        decompress_image(tmp_in_path, tmp_out_path)
                    
                    img = Image.open(tmp_out_path)
                    st.image(img, caption="Reconstructed Image", use_container_width=True)
                    
                    with open(tmp_out_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Reconstructed Image",
                            f.read(),
                            file_name="reconstructed.png",
                            mime="image/png"
                        )
                    
                    cleanup_temp_files(tmp_in_path, tmp_out_path)
                    
                except Exception as e:
                    st.error(f"Decompression failed: {str(e)}")
                    cleanup_temp_files(tmp_in_path, tmp_out_path)

def render_batch_mode(platform, watermark_options, ffmpeg_available):
    """Render batch processing interface"""
    st.header("📦 Batch Processing")
    
    st.warning("Batch processing is currently optimized for images. Video batch processing requires a local installation with full FFmpeg support.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_dir = st.text_input("Input Directory", placeholder="/path/to/input/folder")
        media_type = st.selectbox("File Type", ["image", "video"])
    
    with col2:
        output_dir = st.text_input("Output Directory", placeholder="/path/to/output/folder")
    
    if media_type == "video":
        st.info("For video batch processing, please download and run SoulGenesis locally where you can install FFmpeg with all codecs.")
        return
    
    if st.button("🚀 Process Batch", type="primary", use_container_width=True):
        if not input_dir or not output_dir:
            st.error("Please specify both input and output directories")
            return
        
        if not os.path.exists(input_dir):
            st.error("Input directory does not exist")
            return
        
        if media_type == "video" and not ffmpeg_available:
            st.error("FFmpeg not available. Video compression requires FFmpeg.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Simple batch processing for images only
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_files = [f for f in input_path.iterdir() if f.is_file() and get_file_type(f.name) == "image"]
            
            results = []
            for i, file_path in enumerate(image_files):
                try:
                    config = load_platform_config().get(platform, {}).get("image", {})
                    output_file = output_path / f"compressed_{file_path.name}"
                    
                    method, quality = compress_image(str(file_path), str(output_file), config)
                    results.append({
                        'file': file_path.name,
                        'status': 'success',
                        'method': method,
                        'quality': quality
                    })
                    
                    progress = (i + 1) / len(image_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {progress*100:.1f}%")
                    
                except Exception as e:
                    results.append({
                        'file': file_path.name,
                        'status': 'error',
                        'error': str(e)
                    })
            
            st.session_state.batch_results = results
            status_text.text("Batch processing complete!")
            
            # Show results
            success_count = sum(1 for r in results if r['status'] == 'success')
            error_count = sum(1 for r in results if r['status'] == 'error')
            
            st.success(f"✅ Processed {success_count} files successfully")
            if error_count > 0:
                st.error(f"❌ Failed to process {error_count} files")
            
            # Show detailed results
            with st.expander("View Detailed Results"):
                for result in results:
                    if result['status'] == 'success':
                        st.write(f"✅ {result['file']} ({result['method']}, Q:{result['quality']})")
                    else:
                        st.write(f"❌ {result['file']}: {result['error']}")
        
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")

# -----------------------------
# Run the application
# -----------------------------
if __name__ == "__main__":
    main()