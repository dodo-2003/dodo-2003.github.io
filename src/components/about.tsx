import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Code, Database, Cpu, Zap } from "lucide-react"

const techStack = [
  { category: "AI/ML", items: ["Python", "TensorFlow", "PyTorch", "Hugging Face", "LangChain", "OpenAI API"] },
  { category: "Databases", items: ["MongoDB", "PostgreSQL", "Pinecone", "ChromaDB", "Weaviate", "Redis"] },
  { category: "Backend", items: ["FastAPI", "Node.js", "Express", "Flask", "Docker", "Kubernetes"] },
  { category: "Cloud", items: ["AWS", "Google Cloud", "Azure", "Vercel", "Supabase", "Railway"] }
]

const stats = [
  { icon: Code, label: "Projects Completed", value: "50+", color: "text-ai-blue" },
  { icon: Database, label: "Data Sources Integrated", value: "100+", color: "text-ai-green" },
  { icon: Cpu, label: "Models Trained", value: "25+", color: "text-ai-purple" },
  { icon: Zap, label: "APIs Deployed", value: "75+", color: "text-ai-cyan" }
]

export function About() {
  return (
    <section id="about" className="py-20 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-mono font-bold mb-4 bg-gradient-secondary bg-clip-text text-transparent">
            About Me
          </h2>
          <p className="text-lg text-muted-foreground font-mono max-w-3xl mx-auto">
            Passionate AI engineer with expertise in building intelligent systems that bridge the gap between 
            raw data and meaningful conversations. Specialized in RAG architectures and custom AI solutions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-16">
          <div>
            <h3 className="text-2xl font-mono font-semibold mb-6 text-primary">Technical Expertise</h3>
            <div className="space-y-6">
              {techStack.map((tech, index) => (
                <div key={index}>
                  <h4 className="font-mono font-medium mb-3 text-foreground">{tech.category}</h4>
                  <div className="flex flex-wrap gap-2">
                    {tech.items.map((item, itemIndex) => (
                      <Badge 
                        key={itemIndex} 
                        variant="outline" 
                        className="font-mono text-xs border-primary/30 hover:bg-primary/10 transition-colors"
                      >
                        {item}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-mono font-semibold mb-6 text-primary">Experience Highlights</h3>
            <div className="space-y-4">
              <div className="p-4 rounded-lg border border-border/50 bg-card/50">
                <h4 className="font-mono font-medium mb-2">RAG System Architect</h4>
                <p className="text-sm text-muted-foreground font-mono">
                  Designed and implemented enterprise-scale RAG systems processing millions of documents 
                  with 95%+ accuracy in retrieval and generation tasks.
                </p>
              </div>
              
              <div className="p-4 rounded-lg border border-border/50 bg-card/50">
                <h4 className="font-mono font-medium mb-2">AI Integration Specialist</h4>
                <p className="text-sm text-muted-foreground font-mono">
                  Integrated AI capabilities into existing business workflows, reducing manual processing 
                  time by 80% and improving data accuracy significantly.
                </p>
              </div>
              
              <div className="p-4 rounded-lg border border-border/50 bg-card/50">
                <h4 className="font-mono font-medium mb-2">Custom Model Developer</h4>
                <p className="text-sm text-muted-foreground font-mono">
                  Fine-tuned domain-specific models for healthcare, finance, and legal industries, 
                  achieving state-of-the-art performance on specialized tasks.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, index) => {
            const IconComponent = stat.icon
            return (
              <Card key={index} className="text-center border-border/50 bg-card/50 hover:shadow-card transition-all duration-300">
                <CardContent className="p-6">
                  <IconComponent className={`h-8 w-8 mx-auto mb-3 ${stat.color}`} />
                  <div className="text-2xl font-mono font-bold mb-1">{stat.value}</div>
                  <div className="text-xs font-mono text-muted-foreground">{stat.label}</div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}