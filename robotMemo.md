* 一个可以买到机器人相关零配件的网站：https://www.robotshop.com/jp/ja/robots-for-the-house.html

* [**Blender**](https://www.blender.org/) is the free and open source 3D creation suite. It supports the entirety of the 3D pipeline—modeling, rigging, animation, simulation, rendering, compositing and motion tracking, even video editing and game creation. 

    Blender默认的默认导出格式就是 **.dae**格式的，相比obj文件，dae文件不仅存储了顶点，法向量，面，贴图坐标，材质信息，还存储了整个3D场景的结构信息以及骨骼动画信息。

    官方网址：https://www.blender.org/ <br>
    中文文档：https://docs.blender.org/manual/zh-hans/dev/getting_started/index.html<br>
    非官方中文网址：http://blender.bgteach.com/<br>
    如果要买支持Blender，而且已经制作好的3D模型，可以访问：https://www.turbosquid.com/<br>
    更多的信息可以查询这里：[60个国外免费3D模型下载网站](https://blog.csdn.net/u014581901/article/details/51223036)

    Blender只是提供了机器人的3D建模，可以确定机械构件之间的约束关系，但是并不提供运动仿真功能（那本来也是仿真平台的工作）。

* [**Gazebo**](http://gazebosim.org/)与[**ROS**](https://www.ros.org/)Gazebo offers the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. At your fingertips is a robust physics engine, high-quality graphics, and convenient programmatic and graphical interfaces. Best of all, Gazebo is free with a vibrant community. 

    开源软件，而且与ROS天然结合，如果安装了Full版本的ROS，则不需要另外安装Gazebo，可以为机器人添加现实世界的物理性质, 但是建模的能力稍微弱一些。一些介绍和基本的操作可以看[这里](https://blog.csdn.net/kevin_chan04/article/details/78467218).此外，Gazebo提供了CAD，Blender等各种2D，3D设计软件的接口，可以导入这些图纸让Gazebo的机器人模型更加真实.

* 仿真平台与设计平台之间的关系
    与机器人设计工具（如Solidworks, Blender）不同，机器人仿真平台集成了物理引擎，物理引擎可以根据物体的物理属性计算运动、旋转和碰撞，广泛应用于游戏仿真中。常用的机器人仿真物理引擎有Bullet, ODE, MuJoCo等，其中 **Bullet和ODE开源**，而MuJoCo是商业引擎，Deep Mind本月发布的新的增强学习环境Control Suite中使用的引擎即是MuJoCo。[这篇](https://en.wikipedia.org/wiki/Robotics_simulator)和[这篇](https://blog.csdn.net/ZhangRelay/article/details/42586491)文章对机器人仿真引擎的介绍比较全面好.

    机器人中常用的仿真工具有Gazebo，V-REP，Webot, 其中Webot是商业仿真工具. Gazebo与ROS之间通过通用机器人格式化描述文件[URDF](https://blog.csdn.net/fromcaolei/article/details/50826066)来连接。

* 一个思路：如果不能找到Gazebo现成的模型，可以从[turbosquid](https://www.turbosquid.com/)购买现成的3D模型，通过[Blender](https://www.blender.org/)将模型的.dae文件导出为URDF文件，然后导入Gazebo中，结合ROS来控制。具体看[这里](https://blog.csdn.net/cyril__li/article/details/78881968)
