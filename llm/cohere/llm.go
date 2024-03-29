package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"log"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/cohere"
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
}

var (
	_ llms.LLM = (*LLM)(nil)
	//_ llms.LanguageModel = (*LLM)(nil)
)

var (
	ErrMissingRegion = errors.New("empty region")
)

const cohereCommandModelID = "cohere.command-text-v14" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html

func New(region string) (*LLM, error) {

	if region == "" {
		return nil, ErrMissingRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		return nil, err
	}

	return &LLM{
		brc: bedrockruntime.NewFromConfig(cfg),
	}, nil
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

func (o *LLM) Generate(ctx context.Context, prompts []string, options ...llms.CallOption) ([]*llms.Generation, error) {
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMStart(ctx, prompts)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	payload := cohere.Request{
		Prompt:            prompts[0],
		Temperature:       opts.Temperature,
		P:                 opts.TopP,
		K:                 float64(opts.TopK),
		MaxTokens:         opts.MaxTokens,
		StopSequences:     opts.StopWords,
		ReturnLikelihoods: cohere.None,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	//log.Println("payload\n", string(payloadBytes))

	output, err := o.brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(cohereCommandModelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return nil, err
	}

	var resp cohere.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	generations := []*llms.Generation{
		{Text: resp.Generations[0].Text},
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
