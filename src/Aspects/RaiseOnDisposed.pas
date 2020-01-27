namespace TensorFlow.Island.Aspects;

interface

uses
  RemObjects.Elements.Cirrus,
  RemObjects.Elements.Cirrus.Statements,
  RemObjects.Elements.Cirrus.Values;

type
  [AttributeUsage(AttributeTargets.Class)]
  RaiseOnDisposedAttribute = public class(Attribute, IMethodImplementationDecorator)
  public
    method HandleImplementation(Services: IServices; aMethod: IMethodDefinition);    
  end;

implementation

method RaiseOnDIsposedAttribute.HandleImplementation(Services: IServices; aMethod: IMethodDefinition);
begin
  if String.Equals(aMethod.Name, '.', StringComparison.OrdinalIgnoreCase) or
     String.Equals(aMethod.Name, '~', StringComparison.OrdinalIgnoreCase) or
     (aMethod.Visibility <> Visibility.Public) or aMethod.Static
  then begin
    exit;
  end;

  aMethod.ReplaceMethodBody(
    new BeginStatement(   
      new StandaloneStatement(new ProcValue(new SelfValue, 'CheckAndRaiseOnDisposed')),
      new PlaceHolderStatement)
    ); 
end;

end.