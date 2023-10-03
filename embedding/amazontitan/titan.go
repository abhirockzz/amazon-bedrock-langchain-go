package amazontitan

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"

	titan_embedding "github.com/abhirockzz/amazon-bedrock-go-inference-params/amazontitan/embedding"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/embeddings"
)

type TitanEmbedder struct {
	brc *bedrockruntime.Client

	StripNewLines bool
	BatchSize     int
}

var _ embeddings.Embedder = &TitanEmbedder{}

var (
	ErrMissingRegion = errors.New("empty region")
)

func New(region string) (*TitanEmbedder, error) {

	if region == "" {
		return nil, ErrMissingRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		return nil, err
	}

	return &TitanEmbedder{
		brc: bedrockruntime.NewFromConfig(cfg),
	}, nil

}

func (te *TitanEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {

	log.Println("ENTER TitanEmbedder/EmbedDocuments")

	batchedTexts := embeddings.BatchTexts(
		embeddings.MaybeRemoveNewLines(texts, te.StripNewLines),
		te.BatchSize,
	)

	log.Println("output after creating batched texts")

	for _, text := range batchedTexts {
		fmt.Println(text)
	}

	emb := make([][]float32, 0, len(texts))

	for _, texts := range batchedTexts {
		curTextEmbeddings, err := te.createEmbedding(ctx, texts)
		if err != nil {
			return nil, err
		}

		textLengths := make([]int, 0, len(texts))
		for _, text := range texts {
			textLengths = append(textLengths, len(text))
		}

		combined, err := embeddings.CombineVectors(curTextEmbeddings, textLengths)
		if err != nil {
			return nil, err
		}

		emb = append(emb, combined)
	}

	log.Println("EXIT TitanEmbedder/EmbedDocuments")

	return emb, nil
}

func (te *TitanEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if te.StripNewLines {
		text = strings.ReplaceAll(text, "\n", " ")
	}

	emb, err := te.createEmbedding(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	return emb[0], nil
}

const (
	titanEmbeddingModelID = "amazon.titan-embed-text-v1" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

func (te *TitanEmbedder) createEmbedding(ctx context.Context, texts []string) ([][]float32, error) {

	log.Println("ENTER TitanEmbedder/createEmbedding")

	embeddings := make([][]float32, 0, len(texts))

	var payload titan_embedding.Request

	for _, input := range texts {

		payload = titan_embedding.Request{
			InputText: input,
		}

		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}

		log.Println("creating embedding for", input)

		output, err := te.brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
			Body:        payloadBytes,
			ModelId:     aws.String(titanEmbeddingModelID),
			ContentType: aws.String("application/json"),
		})

		if err != nil {
			return nil, err
		}

		var resp titan_embedding.Response

		err = json.Unmarshal(output.Body, &resp)

		if err != nil {
			return nil, err
		}

		log.Println("finished creating embedding for", input)
		embeddings = append(embeddings, resp.Embedding)
	}

	log.Println("EXIT TitanEmbedder/createEmbedding")

	return embeddings, nil
}
