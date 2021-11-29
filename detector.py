from craft_text_detector import Craft


def detector(image_path, path):
    try:
        try:
            craft = Craft(output_dir=path, crop_type="poly", cuda=False)
        except Exception as e:
            print(e);
        prediction_result = craft.detect_text(image_path)
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()
        return prediction_result['text_crop_paths']
    except Exception as e:
        print(e);