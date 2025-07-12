import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Bot, Database, FileText, Brain, Layers, Code } from "lucide-react"

const services = [
  {
    icon: Bot,
    title: "RAG Systems",
    description: "Retrieval-Augmented Generation systems that combine the power of large language models with your custom knowledge base.",
    features: ["Document Processing", "Vector Embeddings", "Semantic Search", "Real-time Retrieval"],
    gradient: "from-ai-blue to-ai-purple",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-blue)/0.3)]"
  },
  {
    icon: FileText,
    title: "OCR Processing",
    description: "Advanced Optical Character Recognition with AI-powered text extraction and document understanding capabilities.",
    features: ["Multi-format Support", "Layout Preservation", "Handwriting Recognition", "Data Extraction"],
    gradient: "from-ai-green to-ai-cyan",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-green)/0.3)]"
  },
  {
    icon: Database,
    title: "RAG for Databases",
    description: "Intelligent query systems that allow natural language interaction with your SQL databases and data warehouses.",
    features: ["Natural Language Queries", "Schema Understanding", "Query Optimization", "Multi-DB Support"],
    gradient: "from-ai-purple to-ai-blue",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-purple)/0.3)]"
  },
  {
    icon: Layers,
    title: "RAG for MongoDB",
    description: "Specialized NoSQL document retrieval systems optimized for MongoDB collections and complex data structures.",
    features: ["Document Similarity", "Aggregation Pipelines", "Index Optimization", "JSON Schema Parsing"],
    gradient: "from-ai-cyan to-ai-green",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-cyan)/0.3)]"
  },
  {
    icon: Brain,
    title: "Custom Model Training",
    description: "Fine-tuned AI models trained specifically for your domain, data, and business requirements.",
    features: ["Domain Adaptation", "Few-shot Learning", "Transfer Learning", "Model Optimization"],
    gradient: "from-ai-blue to-ai-green",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-blue)/0.3)]"
  },
  {
    icon: Code,
    title: "Integration & Deployment",
    description: "End-to-end implementation services including API development, cloud deployment, and system integration.",
    features: ["REST APIs", "Cloud Deployment", "Monitoring", "Scalability"],
    gradient: "from-ai-purple to-ai-cyan",
    glowColor: "shadow-[0_0_30px_hsl(var(--ai-purple)/0.3)]"
  }
]

export function Services() {
  return (
    <section id="services" className="py-20 relative">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-mono font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
            AI Services Portfolio
          </h2>
          <p className="text-lg text-muted-foreground font-mono max-w-2xl mx-auto">
            Cutting-edge AI solutions designed to transform your data into intelligent conversational experiences
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => {
            const IconComponent = service.icon
            return (
              <Card 
                key={index} 
                className={`group relative border-border/50 bg-card/50 backdrop-blur-sm hover:${service.glowColor} transition-all duration-300 hover:scale-105`}
              >
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${service.gradient} p-3 mb-4 group-hover:shadow-glow transition-all duration-300`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="font-mono text-xl">{service.title}</CardTitle>
                  <CardDescription className="font-mono text-sm">
                    {service.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {service.features.map((feature, featureIndex) => (
                      <Badge 
                        key={featureIndex} 
                        variant="secondary" 
                        className="mr-2 mb-2 font-mono text-xs bg-muted/50 hover:bg-primary/10 transition-colors"
                      >
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}