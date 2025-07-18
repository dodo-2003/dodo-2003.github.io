import { Header } from "@/components/header";
import { Hero } from "@/components/hero";
import { Services } from "@/components/services";
import { CodeShowcase } from "@/components/code-showcase";
import { About } from "@/components/about";
import { Contact } from "@/components/contact";
import { Footer } from "@/components/footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background font-mono">
      <Header />
      <main>
        <Hero />
        <Services />
        <CodeShowcase />
        <About />
        <Contact />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
