---
tags:
  - WebUI
  - Gradio
  - Python
---
# Gradio
[Doc](https://www.gradio.app/docs)

## 综合案例
```python
# 创建 Gradio 界面
with (gr.Blocks(theme="Soft", title="Title") as demo):
    gr.Markdown('# <font color="red"> Demo')

    story_list = [f for f in os.listdir('data') if 'story_' in f]
    story_list.sort()
    storyboard_list = [f for f in os.listdir('data') if 'sb_' in f]
    storyboard_list.sort()
    character_list = [f for f in os.listdir('data') if 'character_' in f]
    character_list.sort()
    motion_list = [f for f in os.listdir('data') if 'motion_' in f]
    motion_list.sort()
    all_list = [f for f in os.listdir('data') if 'all_' in f] + [f for f in os.listdir('data') if 'demo' in f]
    all_list.sort()
    with gr.Tab("Tab1"):
        with gr.Tab("大纲"):
            with gr.Row():
                with gr.Column():
                    actor_box = gr.Textbox(label="角色")
                    gen_story_outline = gr.Button(value="Generate", variant='primary')
                    with gr.Row():
                        outline_select = gr.Dropdown(label="故事大纲缓存", choices=story_list)
                        load_outline = gr.Button(value="Load")

                    with gr.Row():
                        to_storyboard = gr.Button(value="转分镜")
                        to_character = gr.Button(value="转主角形象")
                story_outline_result = gr.Markdown("输入角色生成一个故事")

        with gr.Tab("分镜"):
            with gr.Row():
                with gr.Column():
                    gen_story_board = gr.Button(value="Generate", variant='primary')
                    input_title = gr.Textbox(label="标题")
                    input_outline = gr.Textbox(label="大纲", lines=12)
                    with gr.Row():
                        storyboard_select = gr.Dropdown(label="分镜缓存", choices=storyboard_list)
                        load_board = gr.Button(value="Load")

                    to_motion = gr.Button(value="转位姿")
                storyboard_result = gr.Markdown("输入大纲生成分镜")

        with gr.Tab("位姿"):
            with gr.Row():
                with gr.Column():
                    gen_motion = gr.Button(value="Generate", variant='primary')
                    input_storyboard = gr.Markdown(label="分镜")
                    with gr.Row():
                        motion_select = gr.Dropdown(label="位姿缓存", choices=motion_list)
                        load_motion = gr.Button(value="Load")
                motion_result = gr.Markdown("点击生成位姿")

        with gr.Tab("主角形象"):
            with gr.Row():
                with gr.Column():
                    gen_character = gr.Button(value="Generate", variant='primary')
                    input_outline2 = gr.Textbox(label="大纲", lines=12)
                    with gr.Row():
                        character_select = gr.Dropdown(label="主角缓存", choices=character_list)
                        load_character = gr.Button(value="Load")
                character_result = gr.Markdown("输入大纲生成主角形象")

        with gr.Tab("对话扩充"):
            with gr.Row():
                with gr.Column():
                    gen_chat = gr.Button(value="Generate", variant='primary')
                    input_outline3 = gr.Textbox(label="大纲", lines=12)
                    input_storyboard2 = gr.Markdown(label="分镜")
                    with gr.Row():
                        chat_select = gr.Dropdown(label="对话缓存", choices=character_list)
                        load_chat = gr.Button(value="Load")
                chat_result = gr.Markdown(" ")

        with gr.Tab("调试"):
            with gr.Column():
                input_debug = gr.Markdown(label="分镜")
                with gr.Row():
                    debug_select = gr.Dropdown(label="debug缓存", choices=["debug_01.md"])
                    load_debug = gr.Button(value="Load")
                gen_debug = gr.Button(value="Generate", variant='primary')
            debug_result = gr.Markdown(" ")

    gen_story_outline.click(query_story_outline, actor_box, [story_outline_result, outline_select])
    load_outline.click(load_story_outline, outline_select, [actor_box, story_outline_result])
    to_storyboard.click(copy_title_content, outline_select, [story_outline_result, input_title, input_outline])
    to_character.click(copy_content, story_outline_result, input_outline2)

    gen_story_board.click(query_storyboard, inputs=[input_title, input_outline],
                          outputs=[storyboard_result, storyboard_select])
    load_board.click(load_story_txt, storyboard_select, storyboard_result)
    to_motion.click(copy_content, storyboard_result, input_storyboard)

    gen_motion.click(query_motion, inputs=input_storyboard,
                            outputs=[motion_result, motion_select])
    load_motion.click(load_story_txt, motion_select, motion_result)

    gen_character.click(query_character, inputs=input_outline2,
                            outputs=[character_result, character_select])
    load_character.click(load_story_txt, character_select, character_result)

    # debug
    load_debug.click(load_story_txt, debug_select, input_debug)
    gen_debug.click(query_debug, input_debug, debug_result)

    if True:
        with gr.Tab("Tab2"):
            with gr.Row():
                with gr.Column():
                    scene_prompt_box = gr.Textbox(lines=4, label="Prompt")
                    run = gr.Button(value="Generate", elem_id="txt2poem_run", variant='primary')

                    with gr.Row():
                        slider_x = gr.Slider(0, 200, value=camera_temp2[0], step=5, label="x偏移")
                        slider_y = gr.Slider(0, 250, value=camera_temp2[1], step=5, label="y偏移")
                        slider_z = gr.Slider(0, 200, value=camera_temp2[2], step=5, label="z偏移")

                    with gr.Row():
                        slider_xr = gr.Slider(0, 1800, value=camera_temp2[3], step=10, label="x旋转")
                        slider_yr = gr.Slider(0, 1800, value=camera_temp2[4], step=10, label="y旋转")
                        slider_zr = gr.Slider(0, 1800, value=camera_temp2[5], step=10, label="z旋转")

                player = gr.Image(label="预览图", width=600, height=384)
            scene_analyse_result = gr.Markdown("输入Prompt创建场景")

        scene_prompt_box.change(generate_preview,
                                [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_x.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_y.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_z.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_xr.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_yr.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])
        slider_zr.change(generate_preview,
                        [scene_prompt_box, scene_analyse_result, slider_x, slider_y, slider_z, slider_xr, slider_yr, slider_zr],
                                [scene_analyse_result, player])

    with gr.Tab("Tab3"):
        gr.Interface(
            mov_video_gen,
            gr.Textbox(value="A character is running on a treadmill.", label="Motion prompt"),
            "video"
        )

    gen_all.click(query_all, inputs=user_prompt,
                  outputs=[
                      story_outline_result_f,
                      story_character,
                      story_location,
                      all_sheet,
                      story_sheet_select
                  ])
    load_story_sheet.click(load_story_all,
                           [story_sheet_select, story_character, story_location],
                           [all_sheet, story_character, story_location, modify_storyboard_start])
    modify_run.click(query_modify, [modify_prompt, modify_storyboard_start, all_sheet], all_sheet)

    # 设置按键触发函数
    if enable_render:
        click_event = run.click(generate_scene, scene_prompt_box)

    # 生成器必须要queue函数
    demo.queue()
    # demo.launch()
    demo.launch(server_name="0.0.0.0")
```

```python
# 两个栏的更换内容会引起category变化，category的变化(可能)进一步触发model的变化，然后引起image的变化
def update_model_box(super_category, category, style):
    if "All" == super_category:
        out_df = df_model_info1
    else:
        out_df = df_model_info1[df_model_info1["super-category"] == super_category]

    if "All" != category:
        out_df = out_df[out_df["category"] == category]

    if "All" != style:
        out_df = out_df[out_df["style"] == style]

    model_list = list(out_df['model_id'])
    if model_list is not None:
        return out_df, gr_Dropdown_update(value=model_list[0], choices=model_list)
    return out_df, gr_Dropdown_update(choices=model_list)

# 创建 Gradio 界面
with gr.Blocks(theme="Soft", title="Meta") as demo:
    gr.Markdown('# <font color="red"> Test demo')

    with gr.Tab("Tab1"):
        with gr.Row():
            room_type_box = gr.Dropdown(room_type_list, value="LivingRoom", label="房间类型")
            with gr.Row():
                with gr.Column():
                    random_box = gr.Checkbox(label="随机")
                    top_index = gr.Dropdown([1, 2, 3, 4, 5], value=1, label="Top")
                run = gr.Button(value="生成布局", elem_id="txt2poem_run", variant='primary')
        with gr.Row():
            with gr.Column():
                gr.Markdown("家具偏好")
                with gr.Tab("客厅"):
                    with gr.Row():
                        chair_box = gr.Textbox('0', label="椅子")
                        light_box = gr.Textbox('0', label="灯")
                        plant_box = gr.Textbox('0', label="盆栽")
                    with gr.Row():
                        table_box = gr.Textbox('0', label='桌子')
                        coffe_table_box = gr.Textbox('0', label="茶几")
                        tv_stand_box = gr.Textbox('0', label="电视柜")
                        sofa_box = gr.Textbox('0', label="沙发")
                with gr.Tab("卧室"):
                    bed_box = gr.Textbox('0', label="床")
                with gr.Row():
                    slider_x = gr.Slider(0, 200, value=75, step=5, label="x偏移")
                    slider_y = gr.Slider(0, 200, value=130, step=5, label="z偏移")
                    slider_xr = gr.Slider(0, 1800, value=600, step=5, label="旋转")
            player = gr.Image(width=600, height=384)
        run_log = gr.Markdown("单击【生成布局】开始")

    if show_predict_tab:
        with gr.Tab("Tab2"):
            gr.Interface(fn=predict_image, inputs="image", outputs="text")

    with gr.Tab("Tab3"):
        with gr.Tab("家具"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        super_category_box = gr.Dropdown(["All"] + list(set(df_model_info1['super-category'])), value="All", label="家具类型")
                        category_box = gr.Dropdown(["All"] + list(set(df_model_info1['category'])), value="All", label="详细类别")
                    style_box = gr.Dropdown(["All"] + list(set(df_model_info1['style'])), value="All", label="风格")
                    model_box = gr.Dropdown(["请先选择家具类型"], label="模型")
                    with gr.Row():
                        model_sub_category = gr.Dropdown(["无"], label="修改类别")
                        with gr.Row():
                            next_model = gr.Button(value="Next")
                            label_save = gr.Button(value="保存")
                player2 = gr.Image(width=600, height=384)

            search_result = gr.Dataframe(row_count=(2, "dynamic"), col_count=(len(model_info1[0].keys()), "fixed"),
                                         label="ModelInfo", headers=list(model_info1[0].keys()))

        with gr.Tab("Tab4"):
            with gr.Row():
                layout_select = gr.Dropdown(layout_list, value=layout_list[0], label="布局选择")
                display_layout = gr.Button(value="显示布局", elem_id="layout_search", variant='primary')
            run_log2 = gr.Markdown("单击【显示布局】开始")

        # 设置按键触发函数
        click_event = run.click(generate_layout,
                                inputs=[room_type_box, bed_box, chair_box, light_box, table_box, coffe_table_box,
                                        tv_stand_box, plant_box, sofa_box, slider_x, slider_y, slider_xr,
                                        random_box, top_index],
                                outputs=[player, run_log])

        click_event_search = display_layout.click(search_layout,
                                inputs=layout_select,
                                outputs=run_log2)

        slider_x.change(generate_layout,
                                inputs=[room_type_box, bed_box, chair_box, light_box, table_box, coffe_table_box,
                                        tv_stand_box, slider_x, slider_y, slider_xr, random_box, top_index],
                                outputs=[player, run_log])

        slider_y.change(generate_layout,
                                inputs=[room_type_box, bed_box, chair_box, light_box, table_box, coffe_table_box,
                                        tv_stand_box, slider_x, slider_y, slider_xr, random_box, top_index],
                                outputs=[player, run_log])

        # 当类型或风格修改后，自动修改搜索结果和二级类型列表和Model列表
        super_category_box.change(display_dataframe, [super_category_box, category_box, style_box], [search_result, category_box, model_box])
        style_box.change(display_dataframe, [super_category_box, category_box, style_box], [search_result, category_box, model_box])
        # 当二级类型修改后，自动修改搜索结果和Model列表
        category_box.change(update_model_box, [super_category_box, category_box, style_box], [search_result, model_box])
        #######
        next_model.click(select_next_model,
                         [super_category_box, category_box, style_box, model_box],
                         outputs=model_box)
        # 当Model修改后，自动修改预览图
        model_box.change(update_preview, model_box, [player2, model_sub_category])

        label_save.click(save_model_info,
                         [super_category_box, category_box, style_box, model_box, model_sub_category],
                         outputs=search_result)

    # 生成器必须要queue函数
    demo.queue()
    # demo.launch()
    demo.launch(server_name="0.0.0.0")

```

```python
def fake_gan2():
    images = os.listdir(database_image_path)
    images.sort()
    images = [os.path.join(database_image_path, f) for f in images]
    images = [(f, ''.join(os.path.basename(f).split('.')[:-1])) for f in images]

    return images


def account_model():
    model_type = len(os.listdir(database_image_path))
    model_mesh = len(os.listdir(mesh_path))
    model_GS = len(os.listdir(GS_path))

    return ("## 模型库统计\n\n数量: %d\n\n模型种类: %d\n\n高斯模型数量: %d\n\nMesh模型数量: %d" %
            (model_mesh+model_GS, model_type, model_GS, model_mesh))


def down_load_all():
    gs_fname = "高斯模型.zip"
    mesh_fname = "Mesh模型.zip"

    md_show = "开始打包...\n\n"
    md_show += "正在打包高斯模型 "
    yield md_show, None

    os.system("zip -r -j %s %s" % (gs_fname, GS_path))
    md_show += "完成\n\n正在打包Mesh模型 "
    yield md_show, gs_fname

    os.system("zip -r -j %s %s" % (mesh_fname, mesh_path))
    md_show += "完成\n\n全部完成\n\n请在下方【文件下载】处下载模型包\n\n"
    yield md_show + account_model(), [gs_fname, mesh_fname]


with gr.Blocks(theme="Soft", title="AI Model Gallery") as demo:
    gallery = gr.Gallery(
        label="原图（使用的图片）",
        value=fake_gan2(), elem_id="gallery",
        columns=[4], object_fit="contain", height="auto")
    with gr.Row():
        btn = gr.Button("查看模型", scale=0)
        btn_dl = gr.Button("一键打包下载", scale=0)
    with gr.Row():
        logs = gr.Markdown("(点击任意按钮刷新)\n\n" + account_model())
        with gr.Column(scale=3):
            with gr.Row():
                preview_gs = gr.Video(label="高斯预览（点击【查看模型】预览）")
                preview_mesh = gr.Model3D(label="Mesh预览")
            # file
            output_file = gr.File(label="文件下载")

    btn.click(fake_gan, logs, [logs, output_file, preview_gs, preview_mesh])
    btn_dl.click(down_load_all, None, [logs, output_file])
    gallery.select(on_select, None, logs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=configs["port"])

```
