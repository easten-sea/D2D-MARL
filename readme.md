一、环境安装: pip install -r requirements.txt

二、第三章方法：
    运行main.py（网络结构见MARL.py，日志见文件夹log），通过修改args参数调整配置，如：
        
        1、修改args.alth实现我们的方法（设置为our）或者baseline（DQN和DDPG等）
        2、修改args.device_num修改设备数量
        3、修改args.poisson_rate修改任务生成率

三、第四章方法：
    运行reputation_main.py（网络结构见re_agent.py，日志见文件夹re_log），通过修改args参数调整配置，如：

        1、修改args.malicious调整恶意设备占比
        2、修改args.device_num修改设备数量
        3、修改args.task_arrival_prob修改任务生成率
通过SWAT.py、RTCM.py和RLTCM.py运行baseline

四、第五章方法：
    运行multi_re_main.py（网络结构见re_agent.py，日志见文件夹re_log），通过修改args参数调整配置，如：

        1、修改args.server_num修改服务器数量
        2、修改args.device_num修改单个边缘服务器管理的设备数量
        3、修改args.task_arrival_prob修改任务生成率