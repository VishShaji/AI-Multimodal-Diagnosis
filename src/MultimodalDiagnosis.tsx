import { useState } from 'react'
import { Button } from "./components/ui/button"
import { Input } from "./components/ui/input"
import { Textarea } from "./components/ui/textarea"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card"
import { Label } from "./components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert"
import { Loader2, AlertCircle, FileUp, FileText, Activity, Image as ImageIcon } from 'lucide-react'
type ModelResult = {
  prediction: number[]
}

type PredictionResults = {
  cnn?: ModelResult
  bert?: ModelResult
  lstm?: ModelResult
  vit?: ModelResult
}

export default function MultimodalDiagnosis() {
  const [cnnFile, setCnnFile] = useState<File | null>(null)
  const [bertText, setBertText] = useState('')
  const [lstmFile, setLstmFile] = useState<File | null>(null)
  const [vitFile, setVitFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<PredictionResults>({})

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResults({})

    const modelTypes = ['cnn', 'bert', 'lstm', 'vit']
    const predictions: PredictionResults = {}

    for (const modelType of modelTypes) {
      try {
        let data: string | ArrayBuffer | null = null

        switch (modelType) {
          case 'cnn':
            if (cnnFile) {
              data = await cnnFile.arrayBuffer()
            }
            break
          case 'bert':
            data = bertText
            break
          case 'lstm':
            if (lstmFile) {
              data = await lstmFile.arrayBuffer()
            }
            break
          case 'vit':
            if (vitFile) {
              data = await vitFile.arrayBuffer()
            }
            break
        }

        if (data) {
          const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              data: data instanceof ArrayBuffer ? Array.from(new Uint8Array(data)) : data,
              model_type: modelType,
            }),
          })

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
          }

          const result = await response.json()
          predictions[modelType as keyof PredictionResults] = result
        }
      } catch (err) {
        console.error(`Error in ${modelType} prediction:`, err)
        setError(`An error occurred while processing ${modelType} data. Please try again.`)
      }
    }

    setResults(predictions)
    setIsLoading(false)
  }

  const renderFileUpload = (
    label: string, 
    accept: string, 
    icon: React.ReactNode, 
    file: File | null, 
    setFile: (file: File | null) => void
  ) => (
    <div className="space-y-2">
      <Label htmlFor={label}>{label}</Label>
      <div className="flex items-center space-x-2">
        <Input
          id={label}
          type="file"
          accept={accept}
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
        />
        {icon}
      </div>
      {file && <p className="text-sm text-muted-foreground">File: {file.name}</p>}
    </div>
  )

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6 text-center">AI-Assisted Multimodal Diagnosis System</h1>
      <form onSubmit={handleSubmit} className="space-y-6">
        <Tabs defaultValue="cnn" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="cnn">CNN</TabsTrigger>
            <TabsTrigger value="bert">BERT</TabsTrigger>
            <TabsTrigger value="lstm">LSTM</TabsTrigger>
            <TabsTrigger value="vit">ViT</TabsTrigger>
          </TabsList>
          <TabsContent value="cnn">
            <Card>
              <CardHeader>
                <CardTitle>CNN Model Input</CardTitle>
                <CardDescription>Upload an X-ray or CT scan image for analysis.</CardDescription>
              </CardHeader>
              <CardContent>
                {renderFileUpload("CNN Image", "image/*", <FileUp className="h-6 w-6" />, cnnFile, setCnnFile)}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="bert">
            <Card>
              <CardHeader>
                <CardTitle>BERT Model Input</CardTitle>
                <CardDescription>Enter clinical notes or patient history.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Label htmlFor="bertText">Clinical Notes</Label>
                  <Textarea
                    id="bertText"
                    placeholder="Enter patient's clinical notes or history here..."
                    value={bertText}
                    onChange={(e) => setBertText(e.target.value)}
                    className="min-h-[100px]"
                  />
                  <FileText className="h-6 w-6" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="lstm">
            <Card>
              <CardHeader>
                <CardTitle>LSTM Model Input</CardTitle>
                <CardDescription>Upload a CSV file with time series data (e.g., vital signs).</CardDescription>
              </CardHeader>
              <CardContent>
                {renderFileUpload("LSTM Data", ".csv", <Activity className="h-6 w-6" />, lstmFile, setLstmFile)}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="vit">
            <Card>
              <CardHeader>
                <CardTitle>ViT Model Input</CardTitle>
                <CardDescription>Upload an additional image for Vision Transformer analysis.</CardDescription>
              </CardHeader>
              <CardContent>
                {renderFileUpload("ViT Image", "image/*", <ImageIcon className="h-6 w-6" />, vitFile, setVitFile)}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Button type="submit" className="w-full" disabled={isLoading}>
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing
            </>
          ) : (
            'Predict'
          )}
        </Button>
      </form>

      {error && (
        <Alert className="mt-4 bg-red-100 border-red-400 text-red-700">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )}

      {Object.keys(results).length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>Analysis from all models</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(results).map(([model, result]) => (
                <div key={model} className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2 capitalize">{model} Model</h3>
                  <p>Prediction: {result.prediction.join(', ')}</p>
                </div>
              ))}
            </div>
          </CardContent>
          <CardFooter>
            <p className="text-sm text-muted-foreground">
              Please consult with a healthcare professional for interpretation of these results.
            </p>
          </CardFooter>
        </Card>
      )}
    </div>
  )
}