import { Bot, Heart } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-border/50 bg-background/80 backdrop-blur-md">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-primary rounded flex items-center justify-center">
              <Bot className="h-3 w-3 text-white" />
            </div>
            <span className="font-mono font-semibold">Tajammul.Dev</span>
          </div>
          
          <p className="text-sm text-muted-foreground font-mono flex items-center gap-1">
            Built by Tajammul Iqbal
          </p>
          
          <p className="text-sm text-muted-foreground font-mono">
            © 2025 Tajammul Iqbal - Chatbox - Portfolio
          </p>
        </div>
      </div>
    </footer>
  )
}