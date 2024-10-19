### 睿抗大模型2024CAIP睿抗大模型应用开发赛

#### 使用

##### 使用预构建可执行文件
1. 下载仓库内的 myllm 文件 (仅linux_x86_64)
2. 设置环境变量 `export STANDARD_FONTS=pdf_fonts`
3. 运行 `./myllm --pdf <your pdf path>`

##### 从源码编译运行
1. 使用 rustup 安装 rust `https://rustup.rs/`
2. 克隆仓库 `git clone https://github.com/liyang8246/myllm && cd myllm`
3. 使用 cargo 编译运行 `STANDARD_FONTS=pdf_fonts cargo run --release -- --pdf <your pdf path>`

#### 例子
- mpu6050: 
从源码运行: `STANDARD_FONTS=pdf_fonts cargo run --release -- --pdf mpu6050.pdf`
预构建文件: `STANDARD_FONTS=pdf_fonts ./myllm --pdf mpu6050.pdf`

#### 技术报告
- [技术报告](./report/report.md)