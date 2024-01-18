package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/claude"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func ProcessStreamingOutput(output *bedrockruntime.InvokeModelWithResponseStreamOutput, handler func(ctx context.Context, chunk []byte) error) (claude.Response, error) {

	var combinedResult string
	resp := claude.Response{}

	for event := range output.GetStream().Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:

			//fmt.Println("payload", string(v.Value.Bytes))

			var resp claude.Response
			err := json.NewDecoder(bytes.NewReader(v.Value.Bytes)).Decode(&resp)
			if err != nil {
				return resp, err
			}

			handler(context.Background(), []byte(resp.Completion))
			combinedResult += resp.Completion

		case *types.UnknownUnionMember:
			fmt.Println("unknown tag:", v.Tag)

		default:
			fmt.Println("union is nil or unknown type")
		}
	}

	resp.Completion = combinedResult

	return resp, nil
}
