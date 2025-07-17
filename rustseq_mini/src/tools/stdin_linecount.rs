use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut count = 0u64;
    
    for line_result in reader.lines() {
        match line_result {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("Error at line {}: {}", count, e);
                break;
            }
        }
        if count % 10_000_000 == 0 && count > 0 {
            println!("Read {} lines so far...", count);
        }
    }
    println!("Total lines from stdin: {}", count);
} 