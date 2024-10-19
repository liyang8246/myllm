
use anyhow::{Context, Error as E, Result};
#[allow(unused_imports)]
use candle_core::cpu_backend::CpuDevice;
#[allow(unused_imports)]
use candle_transformers::models::mimi::candle::backend::BackendDevice;
#[allow(unused_imports)]
use candle_transformers::models::mimi::candle::CudaDevice;
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};

use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::{arg, Command};
use hf_hub::{api::sync::Api, Repo, RepoType};
use orca::llm::bert::Bert;
use orca::llm::Embedding;
use orca::{prompts,prompt};
use orca::qdrant::Qdrant;
use orca::record::pdf::Pdf;
use orca::record::Spin;
use tokenizers::Tokenizer;

struct Model(ModelBase);

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        self.0.forward(xs, s)
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    sample_len: usize,
    revision: String,
    tokenizer_file: Option<String>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Args {
    fn new(prompt:String) -> Self {
        Self {
            prompt,
            temperature: Some(0.8),
            top_p: None,
            seed: 10086,
            sample_len: 10000,
            revision: "main".to_string(),
            tokenizer_file: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
    
}

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("My Program")
        .version("1.0")
        .author("Your Name")
        .about("Does awesome things")
        .arg(arg!(--pdf <prompt> "Sets the pdf file path").required(true))
        .get_matches();
    let pdf = matches.get_one::<String>("pdf").unwrap();

    let pdf_records = Pdf::from_file(pdf, false)
        .context("Failed to read PDF file")?
        .spin()
        .context("Failed to process PDF spin")?
        .split(399);

    let collection = std::path::Path::new(pdf)
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("Failed to extract file stem"))?
        .to_string();

    let qdrant = Qdrant::new("http://eromanga.top:6334")?;
    match qdrant.delete_collection(&collection).await {
        Ok(_) => println!("Collection deleted successfully"),
        Err(_) => (),
    }
    qdrant.create_collection(&collection, 384).await?;

    let bert = Bert::new().build_model_and_tokenizer().await?;
    let embeddings = bert.generate_embeddings(prompts!(&pdf_records)).await?;
    qdrant.insert_many(&collection, embeddings.to_vec2()?, pdf_records).await?;

    let query_embedding = bert.generate_embedding(prompt!("I2C Timing")).await?;
    let result = qdrant.search(&collection, query_embedding.to_vec()?.clone(), 8, None).await?;
    let result = result.iter()
    .filter_map(|found_point| {
        found_point.payload.as_ref().map(|payload| {
            serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string())
        })
    }).collect::<String>();
    let prompt_for_model = format!(r#"
    你是一位非常专业的嵌入式软件专家. 你会收到一份有关 I2C 元器件的 Datasheet 中提取的相关摘录.
    然后, 你会尽最大努力如实以 micropython 代码
    的形式告诉我如何对这个元件进行 初始化 以及 读取数据.
    这是 Datasheet 的相关摘要: {}
    现在, 请你开始输出代码
    "#, result);

    let args = Args::new(prompt_for_model);

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = "Qwen/Qwen2.5-1.5B".to_string();
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    // let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    let filenames = vec![repo.get("model.safetensors")?];
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    // let device = Device::Cuda(CudaDevice::new(0).unwrap());
    // let dtype = DType::BF16;
    let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = {
            let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model(ModelBase::new(&config, vb)?)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    println!("----- Model Output -----");
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
