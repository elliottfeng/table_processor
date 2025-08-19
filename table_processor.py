import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps, ImageChops
import base64
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models


# 初始化腾讯云OCR客户端
@st.cache_data
def get_secrets():
    return {
        "TENCENT_SECRET_ID": st.secrets["TENCENT_SECRET_ID"],
        "TENCENT_SECRET_KEY": st.secrets["TENCENT_SECRET_KEY"],
        "TENCENT_REGION": st.secrets.get("TENCENT_REGION", "ap-guangzhou")
    }


def get_ocr_client():
    secrets = get_secrets()
    cred = credential.Credential(secrets["TENCENT_SECRET_ID"], secrets["TENCENT_SECRET_KEY"])
    http_profile = HttpProfile()
    http_profile.endpoint = "ocr.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return ocr_client.OcrClient(cred, secrets["TENCENT_REGION"], client_profile)


def preprocess_image(img, mode="default"):
    """图片预处理函数"""
    try:
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 仅在增强模式下进行裁剪和边框处理
        if mode == "enhanced":
            # 手动实现裁剪白边
            bg = Image.new("RGB", img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            if bbox:
                img = img.crop(bbox)

            # 添加边框
            img = ImageOps.expand(img, border=20, fill='white')

        # 调整尺寸
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))

        return img
    except Exception as e:
        st.error(f"图片预处理失败: {str(e)}")
        return img  # 返回原始图片作为后备


def process_table_data(ocr_data, processing_mode="raw"):
    """处理OCR返回的表格数据"""
    try:
        if not ocr_data.get("TableDetections"):
            return None

        tables = []
        for table in ocr_data["TableDetections"]:
            if not table.get("Cells"):
                continue

            # 计算表格尺寸
            max_row = max(cell["RowTl"] + cell.get("RowSpan", 1) for cell in table["Cells"])
            max_col = max(cell["ColTl"] + cell.get("ColSpan", 1) for cell in table["Cells"])

            # 初始化表格
            grid = [["" for _ in range(max_col)] for _ in range(max_row)]

            # 填充数据
            for cell in table["Cells"]:
                row_start = cell.get("RowTl", 0)
                col_start = cell.get("ColTl", 0)
                row_span = cell.get("RowSpan", 1)
                col_span = cell.get("ColSpan", 1)
                text = cell.get("Text", "")

                # 原始模式不做任何处理，增强模式会去除空行和空列
                for r in range(row_start, min(row_start + row_span, max_row)):
                    for c in range(col_start, min(col_start + col_span, max_col)):
                        grid[r][c] = text

            # 转换为DataFrame
            df = pd.DataFrame(grid)

            # 增强模式下进行后处理
            if processing_mode == "enhanced":
                # 去除全空的行和列
                df = df.dropna(how='all').dropna(axis=1, how='all')
                df = df.reset_index(drop=True)

            tables.append(df)

        return tables[0] if tables else None

    except Exception as e:
        st.error(f"表格处理失败: {str(e)}")
        return None


def process_single_image(img, processing_mode="raw"):
    """处理单张图片"""
    try:
        # 根据模式选择预处理方式
        img = preprocess_image(img, "enhanced" if processing_mode == "enhanced" else "default")

        # Base64编码
        with BytesIO() as buffer:
            img.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            if len(img_base64) < 100:
                raise ValueError("Base64数据异常")

        # OCR请求
        client = get_ocr_client()
        req = models.RecognizeTableAccurateOCRRequest()
        req.from_json_string(json.dumps({
            "ImageBase64": img_base64,
            "TableLanguage": "zh",
            "EnableDetectText": True
        }))

        resp = client.RecognizeTableAccurateOCR(req)
        return process_table_data(json.loads(resp.to_json_string()), processing_mode)

    except Exception as e:
        st.error(f"图片处理失败: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="智能表格识别系统", layout="wide")
    st.title("📊 智能表格识别系统")

    # 模式选择
    processing_mode = st.sidebar.radio(
        "处理模式",
        ["原始模式", "增强模式"],
        index=0,
        help="原始模式: 直接返回OCR结果; 增强模式: 优化图片和表格数据"
    )
    processing_mode = "raw" if processing_mode == "原始模式" else "enhanced"

    # 文件上传
    uploaded_files = st.file_uploader(
        "上传表格图片（支持多选）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 图片预览
        st.subheader("图片预览")
        cols = st.columns(3)
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                cols[i % 3].image(img, caption=file.name, use_container_width=True)
            except:
                cols[i % 3].error("预览失败")

        if st.button("开始识别", type="primary"):
            all_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"正在处理 {i + 1}/{len(uploaded_files)}: {file.name}")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    img = Image.open(file)
                    df = process_single_image(img, processing_mode)

                    if df is not None:
                        df["_来源文件"] = file.name
                        all_data.append(df)
                        st.success(f"{file.name} 识别成功（{len(df)}行）")
                except Exception as e:
                    st.error(f"{file.name} 处理失败: {str(e)}")

            progress_bar.empty()
            status_text.empty()

            if all_data:
                try:
                    final_df = pd.concat(all_data, ignore_index=True)

                    st.subheader("识别结果")
                    st.dataframe(final_df)

                    # 导出Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        final_df.to_excel(writer, index=False)

                    st.download_button(
                        "下载Excel",
                        data=output.getvalue(),
                        file_name="表格识别结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"结果合并失败: {str(e)}")
            else:
                st.error("没有识别到有效数据")


if __name__ == "__main__":
    main()