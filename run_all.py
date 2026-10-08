from data_gen import generate_dataset
from train_mlp import main as train_main
from design_search import main as design_main


if __name__ == '__main__':
    print('Step 1: Generate dataset')
    generate_dataset()

    print('Step 2: Train MLP and run data size study')
    train_main()

    print('Step 3: MLP-assisted design and TMM verification')
    design_main()