graph TD
    %% 定义样式
    classDef dataBox fill:#f9f0ff,stroke:#d8b4e2,stroke-width:2px;
    classDef netBox fill:#e3f2fd,stroke:#90caf9,stroke-width:2px;
    classDef dcBox fill:#ffebee,stroke:#ef9a9a,stroke-width:2px;
    classDef lossBox fill:#e8f5e9,stroke:#a5d6a7,stroke-width:2px;

    subgraph 第一阶段：数据准备与物理初始化 (DataLoader)
        A([读取输入测量数据 Y_in]) ::: dataBox --> B[加载输入系统矩阵 H_in] ::: dataBox
        B --> C[计算 Tikhonov 正则化伪逆 W] ::: dataBox
        C --> D["频域物理初始化<br>(fft -> Y_freq * W -> ifft)"] ::: dcBox
        D --> E([获得初始物理张量 X_init]) ::: dataBox
    end

    subgraph 第二阶段：级联物理感知网络前向传播 (ODFReconModel)
        E --> F["拼接融合 Concat(Y_in, X_init)"] ::: netBox
        F --> G["粗粒度网络 (Coarse 3D UNet)"] ::: netBox
        G --> H([输出 X_coarse]) ::: dataBox
        
        H --> I["第一频域数据一致性层 (DC Layer 1)<br>利用 H_in 修正残差"] ::: dcBox
        I --> J([输出 X_dc_1]) ::: dataBox
        
        J --> K["精细网络 (Refine 3D UNet)"] ::: netBox
        K --> L([输出 X_refined]) ::: dataBox
        
        L --> M["第二频域数据一致性层 (DC Layer 2)<br>利用 H_in 修正残差"] ::: dcBox
        M --> N([输出 X_dc_2]) ::: dataBox
        
        N --> O["抛光网络 (Polish 3D UNet)"] ::: netBox
        O --> P["物理信息引导的密度门控<br>(Density Gate)"] ::: dcBox
        P --> Q([最终预测 ODF 张量 X_final]) ::: dataBox
    end

    subgraph 第三阶段：跨视角自监督计算与优化 (Train)
        Q --> R["利用 H_out 投影得到 Y_out_pred"] ::: dcBox
        Q --> S["利用 H_in 投影得到 Y_in_pred"] ::: dcBox
        
        R & S --> T["计算多维度联合损失<br>(偏振特征/二阶方向矩/空间结构等)"] ::: lossBox
        T --> U(("反向传播与优化<br>(AdamW + AMP)")) ::: lossBox
    end

    %% 阶段间连接
    E -.-> |"输入网络"| F
    Q -.-> |"送入损失计算"| R

  
