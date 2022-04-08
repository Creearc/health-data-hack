import os

output_path = 'results/dataset_2/'
images_output_path = '{}images/'.format(output_path)
annotation_output_path = '{}annotation/'.format(output_path)

def foo(path):
    for file in os.listdir(path):
        name = file.split('.')
        name = '{}.{}'.format(''.join(name[:-1]), name[-1])
        os.rename('{}{}'.format(path, file), '{}{}'.format(path, name))

foo(images_output_path)
foo(annotation_output_path)
