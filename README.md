# Neural Style Transfer

A PyTorch implementation of neural style transfer.

Check out the paper [here](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).


## Quickstart

To run the code quickly:

```sh
pip3 install -r requirements.txt

# run with randomly select content and style images
python3 neural.py

```

## Quick API Usage

```py

if __name__ == '__main__':
    # generate images, create the neural style object
    neural_style_system = NeuralStyle()
    # get randon style+content image
    # neural_style_system.get_img(content_img_name='ma3.jpg',
    #                             style_img_name='flowercarrier.jpg')

    neural_style_system.get_img()

    neural_style_system.plot_content_then_style()

    # select the model (by default, we select the vgg-19 model)
    neural_style_system.select_model(model_selection='vgg19')
    # style weight is how much to prioritize style over content
    # higher style weight = focus more on matching the style picture
    neural_style_system.run_style_transfer(style_weight=1000, content_weight=5)


```

Better documentation coming soon!