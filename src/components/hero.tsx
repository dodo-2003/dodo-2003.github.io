import { Button } from "@/components/ui/button"
import { ArrowDown, Bot, Code2, Sparkles } from "lucide-react"

export function Hero() {
  const scrollToServices = () => {
    document.getElementById('services')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background mesh */}
      <div className="absolute inset-0 bg-gradient-mesh opacity-50" />
      
      {/* Floating pixels animation */}
      <div className="absolute inset-0">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-primary/30 animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      <div className="container mx-auto px-6 text-center relative z-10">
        <div className="mb-8 inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/5">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-sm font-mono text-primary">AI Developer Available</span>
        </div>

        <h1 className="text-4xl md:text-6xl lg:text-7xl font-mono font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
          AI Chatbot
          <br />
          <span className="text-foreground">Engineer</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto font-mono">
          Specialized in <span className="text-primary font-semibold">RAG systems</span>, 
          <span className="text-accent font-semibold"> OCR processing</span>, and 
          <span className="text-ai-purple font-semibold"> custom model training</span>. 
          Building intelligent conversational AI that understands your data.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
          <Button 
            size="lg" 
            className="bg-gradient-primary hover:shadow-glow transition-all duration-300 font-mono"
            onClick={scrollToServices}
          >
            <Bot className="mr-2 h-5 w-5" />
            View My AI Services
          </Button>
          
          <Button 
            variant="outline" 
            size="lg" 
            className="border-primary/50 hover:border-primary hover:bg-primary/5 font-mono"
          >
            <Code2 className="mr-2 h-5 w-5" />
            See Portfolio
          </Button>
        </div>

        <div className="animate-bounce">
          <ArrowDown className="h-6 w-6 text-muted-foreground mx-auto" />
        </div>
      </div>
    </section>
  )
}