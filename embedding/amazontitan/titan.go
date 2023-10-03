package amazontitan

import (
	"context"
	"encoding/json"
	"errors"
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

func (te *TitanEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {

	batchedTexts := embeddings.BatchTexts(
		embeddings.MaybeRemoveNewLines(texts, te.StripNewLines),
		te.BatchSize,
	)

	emb := make([][]float64, 0, len(texts))

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

	return emb, nil
}

func (te *TitanEmbedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
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

func (te *TitanEmbedder) createEmbedding(ctx context.Context, texts []string) ([][]float64, error) {

	//fmt.Println("embeddeding will be calculated for following words")

	embeddings := make([][]float64, 0, len(texts))

	var payload titan_embedding.Request

	for _, input := range texts {

		payload = titan_embedding.Request{
			InputText: input,
		}

		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}

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

		//fmt.Println("response from LLM\n", resp.Embedding)
		embeddings = append(embeddings, resp.Embedding)
	}

	return embeddings, nil
}
