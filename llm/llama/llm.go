package llama

import (
	"context"
	"encoding/json"
	"errors"
	"log"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/llama"
	"github.com/abhirockzz/amazon-bedrock-langchain-go/llm"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

var ErrEmptyResponse = errors.New("empty response")

type LLM struct {
	CallbacksHandler callbacks.Handler
	brc              *bedrockruntime.Client
	modelID          string
}

var (
	_ llms.LLM           = (*LLM)(nil)
	_ llms.LanguageModel = (*LLM)(nil)
)

var (
	ErrMissingRegion = errors.New("empty region")
)

func New(region string, options ...llm.ConfigOption) (*LLM, error) {

	if region == "" {
		return nil, ErrMissingRegion
	}

	llamaLLM := &LLM{modelID: defaultModelID}

	opts := &llm.ConfigOptions{}
	for _, opt := range options {
		opt(opts)
	}

	if opts.BedrockRuntimeClient == nil {
		cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
		if err != nil {
			return nil, err
		}

		llamaLLM.brc = bedrockruntime.NewFromConfig(cfg)
	} else {
		llamaLLM.brc = opts.BedrockRuntimeClient
	}

	if opts.ModelID != "" {
		llamaLLM.modelID = opts.ModelID
	}

	return llamaLLM, nil
}

func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	r, err := o.Generate(ctx, []string{prompt}, options...)
	if err != nil {
		return "", err
	}
	if len(r) == 0 {
		return "", ErrEmptyResponse
	}
	return r[0].Text, nil
}

const (
	defaultModelID = "meta.llama2-13b-chat-v1" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

func (o *LLM) Generate(ctx context.Context, prompts []string, options ...llms.CallOption) ([]*llms.Generation, error) {
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMStart(ctx, prompts)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	payload := llama.Request{
		Prompt:      prompts[0],
		MaxGenLen:   opts.MaxTokens,
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	log.Println("payload\n", string(payloadBytes))

	var resp llama.Response

	if opts.StreamingFunc != nil {

		resp, err = o.invokeAsyncAndGetResponse(payloadBytes, opts.StreamingFunc)
		if err != nil {
			return nil, err
		}

	} else {
		resp, err = o.invokeAndGetResponse(payloadBytes)
		if err != nil {
			return nil, err
		}
	}

	generations := []*llms.Generation{
		{Text: resp.GetResponseString()},
	}

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMEnd(ctx, llms.LLMResult{Generations: [][]*llms.Generation{generations}})
	}
	return generations, nil
}

func (o *LLM) GeneratePrompt(ctx context.Context, prompts []schema.PromptValue, options ...llms.CallOption) (llms.LLMResult, error) {
	return llms.GeneratePrompt(ctx, o, prompts, options...)
}

func (o *LLM) GetNumTokens(text string) int {
	return llms.CountTokens("gpt4", text)
}

func (o *LLM) invokeAndGetResponse(payloadBytes []byte) (llama.Response, error) {

	output, err := o.brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(o.modelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})

	if err != nil {
		return llama.Response{}, err
	}

	var resp llama.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		return llama.Response{}, err
	}

	return resp, nil
}

func (o *LLM) invokeAsyncAndGetResponse(payloadBytes []byte, handler func(ctx context.Context, chunk []byte) error) (llama.Response, error) {

	output, err := o.brc.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        payloadBytes,
		ModelId:     aws.String(o.modelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return llama.Response{}, err
	}

	var resp llama.Response

	resp, err = ProcessStreamingOutput(output, handler)

	if err != nil {
		return llama.Response{}, err
	}

	return resp, nil
}
