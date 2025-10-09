from PIL import Image
#两张图片水平拼接
def merge_images(image1_path, image2_path, output_path):
    # 打开两张图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 获取第一张图片的尺寸
    width1, height1 = image1.size

    # 调整第二张图片的尺寸与第一张相同
    image2 = image2.resize((width1, height1))

    # 创建一个新的空白图片，尺寸为两张图片横向拼接后的尺寸
    merged_image = Image.new('RGB', (width1 * 2, height1))
    # merged_image.save("ttt.jpg")全黑图片

    # 将两张图片拼接到新图片上
    merged_image.paste(image1, (0, 0))#左上角
    merged_image.paste(image2, (width1, 0))
    # 保存合成后的图片
    merged_image.save(output_path)

# merge_images(
#     '/home/zzf/CLIP_WSL/pll_clip/draw/training_accuracy_Caltech101_0.3.png',
#     '/home/zzf/CLIP_WSL/pll_clip/draw/test_accuracy_Caltech101_0.3.png',
#     './merge.png'
#     )

