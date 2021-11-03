from craft_text_detector import Craft


def detector(image_path):
    output_dir = 'detector_outputs/'
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)
    prediction_result = craft.detect_text(image_path)
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    return prediction_result['text_crop_paths']
