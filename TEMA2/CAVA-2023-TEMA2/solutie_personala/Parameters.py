import os

class Parameters:
    def __init__(self):
        self.base_dir = './data'
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemple_pozitive_totale')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exemple_negative_totale')
        self.dir_test_examples = os.path.join(self.base_dir,'exemple_test/testare')
        self.path_annotations = os.path.join(self.base_dir, 'exemple_test/validare/task2_wilma_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 72  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 72  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.use_flip_images = True
        self.number_positive_examples = 3996  # numarul exemplelor pozitive
        self.number_negative_examples = 7890  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0.1
