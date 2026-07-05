use std::process::{Command, Child};
use std::thread;
use std::time::Duration;

fn main() {
    println!("🚀 开始启动联邦学习本地模拟网络...");

    // 1. 启动 Server
    println!("▶️  正在启动 Server...");
    let mut server: Child = Command::new("cargo")
        .args(["run", "--bin", "server"])
        .spawn()
        .expect("无法启动 Server 进程");

    // 给 Server 2 秒钟的时间来绑定 TCP 端口 8888
    thread::sleep(Duration::from_secs(2));

    // 2. 启动 Clients
    let num_clients = 3; // 假设你有 3 个客户端
    let mut clients: Vec<Child> = Vec::new();

    for i in 0..num_clients {
        println!("▶️  正在启动 Client {}...", i);
        let client = Command::new("cargo")
            .args(["run", "--bin", "client", "--", &i.to_string()]) // 传入 client_id
            .spawn()
            .expect(&format!("无法启动 Client {}", i));
        
        clients.push(client);
        // 稍微错开一点启动时间，避免瞬间 CPU 飙升或日志混杂
        thread::sleep(Duration::from_millis(500));
    }

    // 3. 等待 Server 执行完毕（根据你的逻辑，Server 发送 Stop 后会自己退出）
    println!("⏳ 所有节点已启动，等待训练完成...");
    let _ = server.wait().expect("Server 进程异常退出");

    // 4. 清理工作：确保所有 Client 进程都被杀死
    println!("🧹 训练结束，正在清理 Client 进程...");
    for mut client in clients {
        let _ = client.kill();
        let _ = client.wait();
    }

    println!("✅ 模拟运行圆满结束！");
}