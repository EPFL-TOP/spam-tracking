from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QInputDialog, QLabel, QSpinBox
from skimage.util import img_as_float
import h5py
import numpy as np
import napari
import spam.DIC
import math
from cellpose import models, io, core
from tifffile import imread
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)

model = models.CellposeModel(gpu=use_GPU, pretrained_model="/mnt/d/Clement/cellpose_model_membranexnucleus/CP_20241007_h2bxncad")

#_____________________________________________________________________________________
class ProcessingWidget(QWidget):

#_____________________________________________________________________________________
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        self.full_track_data = {}  # Store all track data
        self.image_layer_name = None
        self.selected_track = None
        # Button to open HDF5 file
        self.load_button = QPushButton("Load File")
        self.layout().addWidget(self.load_button)
        self.load_button.clicked.connect(self.open_file_dialog)

        # Button to delete the last selected point
        self.delete_button = QPushButton("Delete Last Point")
        self.layout().addWidget(self.delete_button)
        self.delete_button.clicked.connect(self.on_delete_last_point)

        self.layout().addWidget(QLabel("Tracking Parameters"))

        self.x_box_label = QLabel("X box value:")
        self.x_box_spinbox = QSpinBox()
        self.x_box_spinbox.setRange(1, 100)
        self.x_box_spinbox.setValue(40)

        self.y_box_label = QLabel("Y box value:")
        self.y_box_spinbox = QSpinBox()
        self.y_box_spinbox.setRange(1, 100)
        self.y_box_spinbox.setValue(40)

        self.z_box_label = QLabel("Z box value:")
        self.z_box_spinbox = QSpinBox()
        self.z_box_spinbox.setRange(1, 100)
        self.z_box_spinbox.setValue(10)

        self.layout().addWidget(self.x_box_label)
        self.layout().addWidget(self.x_box_spinbox)
        self.layout().addWidget(self.y_box_label)
        self.layout().addWidget(self.y_box_spinbox)
        self.layout().addWidget(self.z_box_label)
        self.layout().addWidget(self.z_box_spinbox)

        self.forward_button = QPushButton("Track Forward")
        self.backward_button = QPushButton("Track Backward")
        self.layout().addWidget(self.forward_button)
        self.layout().addWidget(self.backward_button)
        self.forward_button.clicked.connect(self.track_forward)
        self.backward_button.clicked.connect(self.track_backward)

        self.cellpose_button = QPushButton("CellPose seg")
        self.layout().addWidget(self.cellpose_button)
        self.cellpose_button.clicked.connect(self.cellpose_seg)


        #to display selected points for tracking
        self.points_layer = self.viewer.add_points(
            [],  # Initially empty
            name="Selected Points",
            ndim=4,  # Specify the 4D space
            size=10,  # Size of the dot
            face_color="red",  # Color of the dot
            border_color="red",
            visible=True,
        )

        #to display the tracked
        self.tracked_points_layer = self.viewer.add_points(
            [],  # Initially empty
            name="Tracked Points",
            ndim=4,  # Specify the 4D space
            size=5,  # Size of the dot
            face_color="blue",  # Color of the dot
            border_color="blue",
            visible=True,
        )

        # Initialize Tracks layer with a minimal valid placeholder
        placeholder_track = np.array([[0, 0, 0, 0, 0]])  # [track_id, time, z, y, x]
        self.tracks_layer = viewer.add_tracks(
            data=placeholder_track,
            name="Tracks",
            #tail_length=10,
            #head_length=5,
        )

        #self.tracked_points = []
        self.current_id = 1  # Unique ID for each trajectory

        # List to store selected points for one tracking stage
        self.selected_points = []

        # Set up the click event callback
        self.viewer.layers[0].mouse_double_click_callbacks.append(self.add_point)

        self.tracked_points_layer.mouse_drag_callbacks.append(self.select_track)
        #self.tracked_points_layer.events.selected_data.connect(self.select_track)
        self.viewer.dims.events.current_step.connect(self.update_z_layer)


#_____________________________________________________________________________________
    def cellpose_seg(self):
        channels = [[1,2]]
        diameter = 25 
        cellprob_threshold = -1
        print('self.viewer.layers ',self.viewer.layers)
        image_layer_data = self.viewer.layers[0].data#.get(self.image_layer_name)
        print('image_layer shape ',image_layer_data.shape)

        for point in self.selected_points:
            
            print(' point ',point)
            image_layer_data_box=image_layer_data[int(point[1]-15):int(point[1]+15), :, int(point[3]-25):int(point[3]+25), int(point[4]-25):int(point[4]+25) ]
            print('image_layer_data_box shape ',image_layer_data_box.shape)
            masks, flows, styles = model.eval(image_layer_data_box, diameter=diameter, channels=channels, 
										    cellprob_threshold=cellprob_threshold, 
										    do_3D=True, anisotropy=1.5, min_size=1000)


            print('masks shape ',masks.shape)
            full_image_mask = np.zeros((image_layer_data.shape[0], image_layer_data.shape[2], image_layer_data.shape[3]), dtype=np.uint8)

            subimage_half_shape = (image_layer_data_box.shape[0] // 2, image_layer_data_box.shape[2] // 2, image_layer_data_box.shape[3] // 2)
            print('subimage_half_shape ', subimage_half_shape)
            # Calculate start and end indices in the full image
            z_start = max(int(point[1]) - subimage_half_shape[0], 0)
            y_start = max(int(point[3]) - subimage_half_shape[1], 0)
            x_start = max(int(point[4]) - subimage_half_shape[2], 0)

            z_end = min(int(point[1]) + subimage_half_shape[0] , image_layer_data.shape[0])
            y_end = min(int(point[3]) + subimage_half_shape[1] , image_layer_data.shape[2])
            x_end = min(int(point[4]) + subimage_half_shape[2] , image_layer_data.shape[3])

            # Calculate the corresponding region in the subimage
            sub_z_start = max(subimage_half_shape[0] - int(point[1]), 0)
            sub_y_start = max(subimage_half_shape[1] - int(point[3]), 0)
            sub_x_start = max(subimage_half_shape[2] - int(point[4]), 0)

            sub_z_end = sub_z_start + (z_end - z_start)
            sub_y_end = sub_y_start + (y_end - y_start)
            sub_x_end = sub_x_start + (x_end - x_start)

            print("Full image slice:", z_start, z_end, y_start, y_end, x_start, x_end)
            print("Subimage slice:", sub_z_start, sub_z_end, sub_y_start, sub_y_end, sub_x_start, sub_x_end)

            # Insert the subimage mask into the full image mask
            print('full_image_mask[z_start:z_end, y_start:y_end, x_start:x_end] shape               ',full_image_mask[z_start:z_end, y_start:y_end, x_start:x_end].shape)
            print('masks[sub_z_start:sub_z_end, sub_y_start:sub_y_end, sub_x_start:sub_x_end] shape ',masks[sub_z_start:sub_z_end, sub_y_start:sub_y_end, sub_x_start:sub_x_end].shape)
            full_image_mask[z_start:z_end, y_start:y_end, x_start:x_end] = \
                masks[sub_z_start:sub_z_end, sub_y_start:sub_y_end, sub_x_start:sub_x_end]


            # Insert the subimage mask into the full image mask
            #full_image_mask[z_start:z_end, y_start:y_end, x_start:x_end] = masks
            full_image_mask = np.expand_dims(full_image_mask, axis=1)
            print('full_image_mask ',full_image_mask.shape)

                    # Add the new 2D+time image to Napari
            self.viewer.add_labels(
                full_image_mask,
                name="cell pose maks",
                #scale=(1, 1),  # Adjust if necessary
                #colormap="gray"
            )

            io.save_masks(
                full_image_mask,
                masks,
                flows,
                "/mnt/d/Clement/testmask",
                channels=channels,
                png=True,  # Save masks as PNGs
                tif=True,  # Save masks as TIFFs
                save_txt=True,  # Save txt outlines for ImageJ
                save_flows=False,  # Save flows as TIFFs
                save_outlines=False,  # Save outlines as TIFFs
                save_mpl=True,  # Make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
            )

#_____________________________________________________________________________________
    def update_z_layer(self, event):
        if self.selected_track==None:return
        current_time = self.viewer.dims.current_step[0]  # Get the current time step
        track = self.full_track_data[self.selected_track]

        for t in track:
            if t[0]==current_time:
                self.viewer.dims.set_point(1, t[1])
                print(f"Updated z-layer to: {t[1]} at time: {current_time}")
                break

#_____________________________________________________________________________________
    def select_track(self, layer, event):
        print('select track ', event )
        if event.type == "mouse_press" and event.button == 1:
            
            print('event pos ',event.position)
            time = self.viewer.dims.current_step[0]
            minDR=9999999999
            for track_id in self.full_track_data:
                for timepoint in self.full_track_data[track_id]:
                    if event.position[0]!=timepoint[0]:continue
                    dR=math.sqrt(math.pow(event.position[1] - timepoint[1], 2) + math.pow(event.position[2] - timepoint[2], 2) + math.pow(event.position[3] - timepoint[3], 2))
                    if dR<minDR: 
                        minDR=dR
                        self.selected_track=track_id

            print('selected track ',self.selected_track)


#_____________________________________________________________________________________
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open HDF5/Tiff File", "", "HDF5/Tiff Files (*.h5 *.hdf5 *.tif *.tiff)")
        file_ext = file_path.split('.')[-1]
        print('file ext ',file_ext)
        if 'h5' == file_ext or 'hdf5' == file_ext:
            self.load_hdf5(file_path)
        if 'tif' == file_ext:
            self.load_tif(file_path)

 
#_____________________________________________________________________________________
    def load_tif(self, file_path):
        print('load tif')
        image_data = imread(file_path)
        self.viewer.add_image(image_data, name="TIFF Image")

#_____________________________________________________________________________________
    def load_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as hdf_file:
            dataset_names = []
            hdf_file.visititems(lambda name, obj: dataset_names.append(name) if isinstance(obj, h5py.Dataset) else None)

        if not dataset_names:
            QMessageBox.warning(self, "No Datasets Found", "This HDF5 file contains no datasets.")
            return

        if len(dataset_names)>1:
            dataset_name, ok = QInputDialog.getItem(self, "Select Dataset", "Dataset:", dataset_names, 0, False)
            if ok and dataset_name:
                self.display_hdf5_data(file_path, dataset_name)
        elif len(dataset_names)==1:
            self.display_hdf5_data(file_path, dataset_names[0])


#_____________________________________________________________________________________
    def display_hdf5_data(self, file_path, dataset_name):
        self.image_layer_name=dataset_name
        with h5py.File(file_path, 'r') as hdf_file:
            data = hdf_file[dataset_name][:]
            self.viewer.add_image(data, name=dataset_name, scale=(1, 1, 1, 1), multiscale=False)


#_____________________________________________________________________________________
    def add_point(self, layer, event):
        time = self.viewer.dims.current_step[0]
        spatial_coords = event.position[-3:]  # (z, y, x)
        coords = np.array([time, *spatial_coords])

        if len(self.points_layer.data) > 0:
            self.points_layer.data = np.vstack([self.points_layer.data, coords])
        else:
            self.points_layer.data = np.array([coords])

        self.selected_points.append(np.array([self.current_id, *coords]))
        self.current_id += 1


#_____________________________________________________________________________________
    def on_delete_last_point(self):
        if self.selected_points:
            self.selected_points.pop()

            if self.selected_points:
                self.points_layer.data = np.array([t[1:] for t in self.selected_points]  )
            else:
                self.points_layer.data = np.empty((0, 4))

#_____________________________________________________________________________________
    def track_backward(self):
        print('track BKW')
        self.spam_tracking(-1)

#_____________________________________________________________________________________
    def track_forward(self):
        print('track FWD')
        self.spam_tracking(1)

#_____________________________________________________________________________________
    def spam_tracking(self, way):
        print('len layers ',len(self.viewer.layers))
        print('self.viewer.layers ',self.viewer.layers)
        print('self.viewer.layers[0] ',self.viewer.layers[0])
        print('self.viewer.layers[0].name ',self.viewer.layers[0].name)

        image_layer_data = self.viewer.layers[0].data#.get(self.image_layer_name)
        print('image_layer type ',type(image_layer_data))
        print('image_layer shape ',image_layer_data.shape)
        image_layer_data = None
        for layer in self.viewer.layers:
            if layer.name == self.image_layer_name:
                image_layer_data = layer.data

        for point in self.selected_points:
            print('track FDW, point ',point)
            self.tracked_points_layer.data = np.vstack([self.tracked_points_layer.data, point[1:]])
            track_id = point[0]
            current_time, z, y, x = point[1:]
            num_time_points = int((self.viewer.dims.range[0].stop - self.viewer.dims.range[0].start) / self.viewer.dims.range[0].step + 1)      

            try:
                self.full_track_data[int(track_id)].append(point[1:])
            except KeyError:
                self.full_track_data[int(track_id)]=[point[1:]]

            dx=self.x_box_spinbox.value()
            dy=self.y_box_spinbox.value()
            dz=self.z_box_spinbox.value()
            print('dx, dy, dz ',dx,' ',dy,' ',dz)
            reg=None
            if way>0:
                for t in range(int(current_time) + 1, num_time_points):

                    print('reg ',reg)
                    reg = spam.DIC.register(
                        image_layer_data[t-1,int(z-dz):int(z+dz), int(y-dy):int(y+dy), int(x-dx):int(x+dx)],
                        image_layer_data[t,  int(z-dz):int(z+dz), int(y-dy):int(y+dy), int(x-dx):int(x+dx)],
                        imShowProgress=0,
                        verbose=0,
                        im1mask=spam.mesh.structuringElement(radius=dx, dim=3),
                        PhiRigid=True
                        )

                    print('reg[Phi][0:3,-1] ',reg['Phi'][0:3,-1])
                    new_point = np.array([t, int(z+reg['Phi'][0:3,-1][0]) , int(y+reg['Phi'][0:3,-1][1]), int(x+reg['Phi'][0:3,-1][2])])
                    self.tracked_points_layer.data = np.vstack([self.tracked_points_layer.data, new_point])
                    new_track = np.array([track_id, *new_point])
                    self.full_track_data[int(track_id)].append(new_point)

            else:
                for t in range(int(current_time) - 1, -1, -1):
                    reg = spam.DIC.register(
                        image_layer_data[t+1,int(z-dz):int(z+dz), int(y-dy):int(y+dy), int(x-dx):int(x+dx)],
                        image_layer_data[t  ,int(z-dz):int(z+dz), int(y-dy):int(y+dy), int(x-dx):int(x+dx)],
                        imShowProgress=0,
                        verbose=0,
                        im1mask=spam.mesh.structuringElement(radius=dx, dim=3),
                        PhiRigid=True
                        )
                    print('reg[Phi][0:3,-1] ',reg['Phi'][0:3,-1])
                    new_point = np.array([t, int(z+reg['Phi'][0:3,-1][0]) , int(y+reg['Phi'][0:3,-1][1]), int(x+reg['Phi'][0:3,-1][2])])
                    self.tracked_points_layer.data = np.vstack([self.tracked_points_layer.data, new_point])
                    new_track = np.array([track_id, *new_point])
                    self.full_track_data[int(track_id)].append(new_point)
            #self.create_2d_time_image(track_id)

        self.update_tracks()
        self.selected_points = []
        self.points_layer.data = []#np.empty((0, 4))


#_____________________________________________________________________________________
    def update_tracks(self, event=None):

        tracks = []

        for track_id in self.full_track_data:
            track_array = np.zeros((len(self.full_track_data[track_id]), 10), dtype=np.float32)
            track = np.array(self.full_track_data[track_id])
            time = track[:, 0]
            z = track[:, 1]
            y = track[:, 2]
            x = track[:, 3]

            gz = np.gradient(z)
            gy = np.gradient(y)
            gx = np.gradient(x)

            speed = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            track_array[:, 0] = track_id
            track_array[:, 1] = time
            track_array[:, 2] = z
            track_array[:, 3] = y
            track_array[:, 4] = x
            track_array[:, 5] = gz
            track_array[:, 6] = gy
            track_array[:, 7] = gx
            track_array[:, 8] = speed
            track_array[:, 9] = distance

            tracks.append(track_array)

        tracks = np.concatenate(tracks, axis=0)
        data = tracks[:, :5]  # just the coordinate data

        features = {
            'time': tracks[:, 1],
            'gradient_z': tracks[:, 5],
            'gradient_y': tracks[:, 6],
            'gradient_x': tracks[:, 7],
            'speed': tracks[:, 8],
            'distance': tracks[:, 9],
        }

        self.tracks_layer.data = data
        self.tracks_layer.features = features
        self.tracks_layer.tail_width = 10


 

    def create_2d_time_image(self, track_id):

        tracked_points = self.full_track_data[track_id]
        image_layer_data = None
        for layer in self.viewer.layers:
            if layer.name == self.image_layer_name:
                image_layer_data = layer.data

        # Create an empty array for the 2D+time image
        time_points = len(tracked_points)
        y_dim, x_dim = image_layer_data.shape[2:]  # Assume all z-planes have the same shape
        new_image_data = np.zeros((time_points, y_dim, x_dim), dtype=image_layer_data.dtype)

        # Extract the z-plane for each tracked point
        for t, (time, z, y, x) in enumerate(tracked_points):
            print('t time, z, y, x ',t,' ', time,' ', z,' ', y,' ', x)
            if time < image_layer_data.shape[0] and z < image_layer_data.shape[1]:
                print('image_layer_data shape ',image_layer_data.shape)
                new_image_data[t] = image_layer_data[int(time), int(z)]
            else:
                print(f"Tracked point out of range at time {time}, z {z}")

        # Add the new 2D+time image to Napari
        self.viewer.add_image(
            new_image_data,
            name=f"Tracked Z-Plane {track_id}",
            scale=(1, 1, 1),  # Adjust if necessary
            colormap="gray"
        )


