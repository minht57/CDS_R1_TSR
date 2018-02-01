### CDS_R2_LD
**This program is: Lane Detection, a part of CDS project.**
### Dependency
- OpenCV 3.2.0
### Compile
1. Clone [`CDS_R1_TSR`](https://github.com/minht57/CDS_R1_TSR) with branch [`LaneDetection`](https://github.com/minht57/CDS_R1_TSR/tree/LaneDetection) from Github to your local machine:
    ```bash
    git clone -b LaneDetection https://github.com/minht57/CDS_R1_TSR.git
    ```
2. Create build folder and compile
    ```bash
    cd YOUR_DIRECTION_PATH/CDS_R1_TSR
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ```
### Test
- Go to CDS_R1_TSR folder
    ```bash
    cd YOUR_DIRECTION_PATH/CDS_R1_TSR
    ```
- Help command
    ```bash
    ./build/src-ld -h
    or 
    ./build/src-ld --help
    ```
- Run with **Image**
    ```bash
    ./build/src-ld -i=test2.jpg
    ```
- Run with **Video**
    - Get video to test:[`highway45.mp4`](https://www.youtube.com/watch?v=T6c0o7iR2u4): https://www.youtube.com/watch?v=T6c0o7iR2u4
    - Test with a frame of video
        ```bash
        ./build/src-ld -v=highway45.mp4 -f=<frame_number>
        ```
        **Example** *(frame_number = 2000)*
        ```bash
        ./build/src-ld -v=highway45.mp4 -f=2000
        ```
    - Test with video
        ```bash
        ./build/src-ld -v=<video_path>
        ```
        **Example**
        ```bash
        ./build/src-ld -v=highway45.mp4
        ```
    - Test with video start at *frame_number*
        ```bash
        ./build/src-ld -v=highway45.mp4 -s=<frame_number>
        ```
        **Example** *(frame_number = 2000)*
        ```bash
        ./build/src-ld -v=highway45.mp4 -s=2000
        ```
